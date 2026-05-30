//! AST Extractor for Rust codebases
//! Outputs JSON in the format expected by the FIM datagen script

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use syn::spanned::Spanned;
use syn::{visit::Visit, Item};
use walkdir::WalkDir;

#[derive(Debug, Serialize, Deserialize)]
struct FileData {
    code: String,
    ast: Vec<AstNode>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum AstNode {
    Mod(ModNode),
    Use(UseNode),
    Struct(StructNode),
    Enum(EnumNode),
    Function(FunctionNode),
    Impl(ImplNode),
    Trait(TraitNode),
    Const(ConstNode),
    Other(OtherNode),
}

#[derive(Debug, Serialize, Deserialize)]
struct ModNode {
    #[serde(rename = "Mod")]
    data: NodeData,
}

#[derive(Debug, Serialize, Deserialize)]
struct UseNode {
    #[serde(rename = "Use")]
    data: NodeData,
}

#[derive(Debug, Serialize, Deserialize)]
struct StructNode {
    #[serde(rename = "Struct")]
    data: NodeData,
}

#[derive(Debug, Serialize, Deserialize)]
struct EnumNode {
    #[serde(rename = "Enum")]
    data: NodeData,
}

#[derive(Debug, Serialize, Deserialize)]
struct FunctionNode {
    #[serde(rename = "Function")]
    data: FunctionData,
}

#[derive(Debug, Serialize, Deserialize)]
struct ImplNode {
    #[serde(rename = "Impl")]
    data: NodeData,
}

#[derive(Debug, Serialize, Deserialize)]
struct TraitNode {
    #[serde(rename = "Trait")]
    data: NodeData,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConstNode {
    #[serde(rename = "Const")]
    data: NodeData,
}

#[derive(Debug, Serialize, Deserialize)]
struct OtherNode {
    #[serde(rename = "Other")]
    data: NodeData,
}

#[derive(Debug, Serialize, Deserialize)]
struct NodeData {
    name: String,
    span: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct FunctionData {
    name: String,
    span: String,
    body: String,
    params: Vec<(String, String)>,
    return_type: String,
}

#[allow(dead_code)]
struct AstVisitor<'a> {
    source: &'a str,
    nodes: Vec<AstNode>,
}

impl<'a> AstVisitor<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            nodes: Vec::new(),
        }
    }

    #[allow(dead_code)]
    fn span_to_string(&self, span: proc_macro2::Span) -> String {
        let start = span.start();
        let end = span.end();
        format!("{}:{}", start.line, end.line)
    }

    #[allow(dead_code)]
    fn extract_span_text(&self, span: proc_macro2::Span) -> String {
        let lines: Vec<&str> = self.source.lines().collect();
        let start = span.start();
        let end = span.end();

        if start.line == 0 || end.line == 0 || start.line > lines.len() {
            return String::new();
        }

        let start_line = start.line.saturating_sub(1);
        let end_line = end.line.min(lines.len());

        lines[start_line..end_line].join("\n")
    }
}

impl<'ast> Visit<'ast> for AstVisitor<'ast> {
    fn visit_item(&mut self, item: &'ast Item) {
        match item {
            Item::Mod(m) => {
                let span = self.span_to_string(m.span());
                self.nodes.push(AstNode::Mod(ModNode {
                    data: NodeData {
                        name: m.ident.to_string(),
                        span,
                    },
                }));
            }
            Item::Use(u) => {
                let span = self.span_to_string(u.span());
                let name = quote::quote!(#u).to_string();
                self.nodes.push(AstNode::Use(UseNode {
                    data: NodeData {
                        name: name.chars().take(100).collect(),
                        span,
                    },
                }));
            }
            Item::Struct(s) => {
                let span = self.span_to_string(s.span());
                self.nodes.push(AstNode::Struct(StructNode {
                    data: NodeData {
                        name: s.ident.to_string(),
                        span,
                    },
                }));
            }
            Item::Enum(e) => {
                let span = self.span_to_string(e.span());
                self.nodes.push(AstNode::Enum(EnumNode {
                    data: NodeData {
                        name: e.ident.to_string(),
                        span,
                    },
                }));
            }
            Item::Fn(f) => {
                // Get full function span (from fn keyword to closing brace)
                let start_line = f.sig.fn_token.span.start().line;
                let end_line = f.block.brace_token.span.close().end().line;
                let span_str = format!("{}:{}", start_line, end_line);

                // Extract function body
                let body = f
                    .block
                    .stmts
                    .iter()
                    .map(|stmt| quote::quote!(#stmt).to_string())
                    .collect::<Vec<_>>()
                    .join(" ");

                // Extract parameters
                let params: Vec<(String, String)> = f
                    .sig
                    .inputs
                    .iter()
                    .filter_map(|arg| match arg {
                        syn::FnArg::Typed(pat_type) => {
                            let name = quote::quote!(#pat_type.pat).to_string();
                            let ty = quote::quote!(#pat_type.ty).to_string();
                            Some((name, ty))
                        }
                        syn::FnArg::Receiver(r) => {
                            Some(("self".to_string(), quote::quote!(#r).to_string()))
                        }
                    })
                    .collect();

                // Extract return type
                let return_type = match &f.sig.output {
                    syn::ReturnType::Default => "()".to_string(),
                    syn::ReturnType::Type(_, ty) => quote::quote!(#ty).to_string(),
                };

                self.nodes.push(AstNode::Function(FunctionNode {
                    data: FunctionData {
                        name: f.sig.ident.to_string(),
                        span: span_str,
                        body: format!("{{ {} }}", body),
                        params,
                        return_type,
                    },
                }));
            }
            Item::Impl(i) => {
                let name = i.self_ty.as_ref();
                let name_str = quote::quote!(#name).to_string();
                let span = self.span_to_string(i.span());
                self.nodes.push(AstNode::Impl(ImplNode {
                    data: NodeData {
                        name: name_str,
                        span,
                    },
                }));

                // Visit methods inside impl
                for item in &i.items {
                    if let syn::ImplItem::Fn(method) = item {
                        // Get full method span
                        let start_line = method.sig.fn_token.span.start().line;
                        let end_line = method.block.brace_token.span.close().end().line;
                        let span_str = format!("{}:{}", start_line, end_line);

                        let body = method
                            .block
                            .stmts
                            .iter()
                            .map(|stmt| quote::quote!(#stmt).to_string())
                            .collect::<Vec<_>>()
                            .join(" ");

                        let params: Vec<(String, String)> = method
                            .sig
                            .inputs
                            .iter()
                            .filter_map(|arg| match arg {
                                syn::FnArg::Typed(pat_type) => {
                                    let name = quote::quote!(#pat_type.pat).to_string();
                                    let ty = quote::quote!(#pat_type.ty).to_string();
                                    Some((name, ty))
                                }
                                syn::FnArg::Receiver(r) => {
                                    Some(("self".to_string(), quote::quote!(#r).to_string()))
                                }
                            })
                            .collect();

                        let return_type = match &method.sig.output {
                            syn::ReturnType::Default => "()".to_string(),
                            syn::ReturnType::Type(_, ty) => quote::quote!(#ty).to_string(),
                        };

                        self.nodes.push(AstNode::Function(FunctionNode {
                            data: FunctionData {
                                name: method.sig.ident.to_string(),
                                span: span_str,
                                body: format!("{{ {} }}", body),
                                params,
                                return_type,
                            },
                        }));
                    }
                }
            }
            Item::Trait(t) => {
                let span = self.span_to_string(t.span());
                self.nodes.push(AstNode::Trait(TraitNode {
                    data: NodeData {
                        name: t.ident.to_string(),
                        span,
                    },
                }));
            }
            Item::Const(c) => {
                let span = self.span_to_string(c.span());
                self.nodes.push(AstNode::Const(ConstNode {
                    data: NodeData {
                        name: c.ident.to_string(),
                        span,
                    },
                }));
            }
            Item::Static(s) => {
                let span = self.span_to_string(s.span());
                self.nodes.push(AstNode::Const(ConstNode {
                    data: NodeData {
                        name: s.ident.to_string(),
                        span,
                    },
                }));
            }
            Item::Type(t) => {
                let span = self.span_to_string(t.span());
                self.nodes.push(AstNode::Other(OtherNode {
                    data: NodeData {
                        name: t.ident.to_string(),
                        span,
                    },
                }));
            }
            _ => {}
        }

        syn::visit::visit_item(self, item);
    }
}

fn process_file(path: &Path, repo_root: &Path) -> Option<(String, FileData)> {
    let content = fs::read_to_string(path).ok()?;

    // Parse the file
    let syntax = syn::parse_file(&content).ok()?;

    // Extract AST nodes
    let mut visitor = AstVisitor::new(&content);
    for item in &syntax.items {
        visitor.visit_item(item);
    }

    // Skip files with no useful nodes
    if visitor.nodes.is_empty() {
        return None;
    }

    // Get relative path
    let rel_path = path.strip_prefix(repo_root).ok()?;
    let rel_path_str = rel_path.to_string_lossy().to_string();

    Some((
        rel_path_str,
        FileData {
            code: content.clone(),
            ast: visitor.nodes,
        },
    ))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: ast_extractor <repo_path> <output_json>");
        eprintln!("Example: ast_extractor /tmp/reth ./data/reth_ast.json");
        std::process::exit(1);
    }

    let repo_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);

    if !repo_path.exists() {
        eprintln!("Error: repo path does not exist: {}", repo_path.display());
        std::process::exit(1);
    }

    // Collect all Rust files
    println!("Scanning for Rust files in {}...", repo_path.display());
    let rust_files: Vec<PathBuf> = WalkDir::new(&repo_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            path.extension().map_or(false, |ext| ext == "rs")
                && !path.to_string_lossy().contains("/target/")
                && !path.to_string_lossy().contains("/.git/")
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    println!("Found {} Rust files", rust_files.len());

    // Progress bar
    let pb = ProgressBar::new(rust_files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    // Process files in parallel
    let results: Mutex<HashMap<String, FileData>> = Mutex::new(HashMap::new());
    let errors: Mutex<usize> = Mutex::new(0);

    rust_files.par_iter().for_each(|path| {
        if let Some((rel_path, data)) = process_file(path, &repo_path) {
            results.lock().unwrap().insert(rel_path, data);
        } else {
            *errors.lock().unwrap() += 1;
        }
        pb.inc(1);
    });

    pb.finish_with_message("done");

    let results = results.into_inner().unwrap();
    let errors = errors.into_inner().unwrap();

    println!(
        "\nProcessed: {} files successfully, {} files skipped/errored",
        results.len(),
        errors
    );

    // Count nodes
    let mut node_counts: HashMap<&str, usize> = HashMap::new();
    for data in results.values() {
        for node in &data.ast {
            let kind = match node {
                AstNode::Mod(_) => "Mod",
                AstNode::Use(_) => "Use",
                AstNode::Struct(_) => "Struct",
                AstNode::Enum(_) => "Enum",
                AstNode::Function(_) => "Function",
                AstNode::Impl(_) => "Impl",
                AstNode::Trait(_) => "Trait",
                AstNode::Const(_) => "Const",
                AstNode::Other(_) => "Other",
            };
            *node_counts.entry(kind).or_default() += 1;
        }
    }

    println!("\nNode counts:");
    let mut counts: Vec<_> = node_counts.iter().collect();
    counts.sort_by(|a, b| b.1.cmp(a.1));
    for (kind, count) in counts {
        println!("  {}: {}", kind, count);
    }

    // Write output
    println!("\nWriting to {}...", output_path.display());
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).ok();
    }

    let json = serde_json::to_string(&results).expect("Failed to serialize");
    fs::write(&output_path, json).expect("Failed to write output");

    let size = fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "✅ Done! Output: {} ({:.1} MB)",
        output_path.display(),
        size as f64 / 1024.0 / 1024.0
    );
}
