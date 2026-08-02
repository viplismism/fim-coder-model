//! Streaming AST Extractor for Rust codebases
//! Emits NDJSON (one JSON object per line) to stdout or a file.
//! Uses `ignore` for fast parallel file walking and `rayon` for processing.

use ignore::WalkBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use syn::spanned::Spanned;
use syn::{visit::Visit, Item};

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
        Self { source, nodes: Vec::new() }
    }

    #[allow(dead_code)]
    fn span_to_string(&self, span: proc_macro2::Span) -> String {
        let start = span.start();
        let end = span.end();
        format!("{}:{}", start.line, end.line)
    }
}

impl<'ast> Visit<'ast> for AstVisitor<'ast> {
    fn visit_item(&mut self, item: &'ast Item) {
        match item {
            Item::Mod(m) => {
                let span = self.span_to_string(m.span());
                self.nodes.push(AstNode::Mod(ModNode { data: NodeData { name: m.ident.to_string(), span } }));
            }
            Item::Use(u) => {
                let span = self.span_to_string(u.span());
                let name = quote::quote!(#u).to_string();
                self.nodes.push(AstNode::Use(UseNode { data: NodeData { name: name.chars().take(100).collect(), span } }));
            }
            Item::Struct(s) => {
                let span = self.span_to_string(s.span());
                self.nodes.push(AstNode::Struct(StructNode { data: NodeData { name: s.ident.to_string(), span } }));
            }
            Item::Enum(e) => {
                let span = self.span_to_string(e.span());
                self.nodes.push(AstNode::Enum(EnumNode { data: NodeData { name: e.ident.to_string(), span } }));
            }
            Item::Fn(f) => {
                let start_line = f.sig.fn_token.span.start().line;
                let end_line = f.block.brace_token.span.close().end().line;
                let span_str = format!("{}:{}", start_line, end_line);
                let body = f.block.stmts.iter().map(|stmt| quote::quote!(#stmt).to_string()).collect::<Vec<_>>().join(" ");
                let params = f.sig.inputs.iter().filter_map(|arg| match arg {
                    syn::FnArg::Typed(p) => {
                        let name = quote::quote!(#p.pat).to_string();
                        let ty = quote::quote!(#p.ty).to_string();
                        Some((name, ty))
                    }
                    syn::FnArg::Receiver(r) => Some(("self".to_string(), quote::quote!(#r).to_string())),
                }).collect();
                let return_type = match &f.sig.output {
                    syn::ReturnType::Default => "()".to_string(),
                    syn::ReturnType::Type(_, ty) => quote::quote!(#ty).to_string(),
                };
                self.nodes.push(AstNode::Function(FunctionNode { data: FunctionData { name: f.sig.ident.to_string(), span: span_str, body: format!("{{ {} }}", body), params, return_type } }));
            }
            Item::Impl(i) => {
                let name = quote::quote!(&i.self_ty).to_string();
                let span = self.span_to_string(i.span());
                self.nodes.push(AstNode::Impl(ImplNode { data: NodeData { name, span } }));
                for it in &i.items {
                    if let syn::ImplItem::Fn(m) = it {
                        let start_line = m.sig.fn_token.span.start().line;
                        let end_line = m.block.brace_token.span.close().end().line;
                        let span_str = format!("{}:{}", start_line, end_line);
                        let body = m.block.stmts.iter().map(|stmt| quote::quote!(#stmt).to_string()).collect::<Vec<_>>().join(" ");
                        let params = m.sig.inputs.iter().filter_map(|arg| match arg {
                            syn::FnArg::Typed(p) => {
                                let name = quote::quote!(#p.pat).to_string();
                                let ty = quote::quote!(#p.ty).to_string();
                                Some((name, ty))
                            }
                            syn::FnArg::Receiver(r) => Some(("self".to_string(), quote::quote!(#r).to_string())),
                        }).collect();
                        let return_type = match &m.sig.output {
                            syn::ReturnType::Default => "()".to_string(),
                            syn::ReturnType::Type(_, ty) => quote::quote!(#ty).to_string(),
                        };
                        self.nodes.push(AstNode::Function(FunctionNode { data: FunctionData { name: m.sig.ident.to_string(), span: span_str, body: format!("{{ {} }}", body), params, return_type } }));
                    }
                }
            }
            Item::Trait(t) => {
                let span = self.span_to_string(t.span());
                self.nodes.push(AstNode::Trait(TraitNode { data: NodeData { name: t.ident.to_string(), span } }));
            }
            Item::Const(c) => {
                let span = self.span_to_string(c.span());
                self.nodes.push(AstNode::Const(ConstNode { data: NodeData { name: c.ident.to_string(), span } }));
            }
            Item::Static(s) => {
                let span = self.span_to_string(s.span());
                self.nodes.push(AstNode::Const(ConstNode { data: NodeData { name: s.ident.to_string(), span } }));
            }
            Item::Type(t) => {
                let span = self.span_to_string(t.span());
                self.nodes.push(AstNode::Other(OtherNode { data: NodeData { name: t.ident.to_string(), span } }));
            }
            _ => {}
        }
        syn::visit::visit_item(self, item);
    }
}

fn process_file(path: &Path, repo_root: &Path) -> Option<(String, FileData)> {
    let content = fs::read_to_string(path).ok()?;
    let syntax = syn::parse_file(&content).ok()?;
    let mut visitor = AstVisitor::new(&content);
    for item in &syntax.items { visitor.visit_item(item); }
    if visitor.nodes.is_empty() { return None; }
    let rel = path.strip_prefix(repo_root).ok()?.to_string_lossy().to_string();
    Some((rel, FileData { code: content, ast: visitor.nodes }))
}

fn write_ndjson<W: Write>(w: &mut W, path: &str, data: &FileData) -> io::Result<()> {
    let ast_json = serde_json::to_string(&data.ast).unwrap_or_else(|_| "[]".to_string());
    writeln!(w, "{{\"path\":{:?},\"code\":{:?},\"ast\":{}}}", path, data.code, ast_json)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: ast_extractor <repo_path> <output_path|->");
        eprintln!("Use '-' for stdout.");
        std::process::exit(1);
    }
    let repo_path = PathBuf::from(&args[1]);
    let out_path = PathBuf::from(&args[2]);
    if !repo_path.is_dir() { eprintln!("Repo path invalid"); std::process::exit(1); }
    let stdout_mode = out_path == Path::new("-");
    let mut writer: Box<dyn Write> = if stdout_mode { Box::new(io::stdout()) } else { Box::new(BufWriter::new(File::create(&out_path).expect("create output"))) };

    let walker = WalkBuilder::new(&repo_path)
        .standard_filters(true)
        .follow_links(true)
        .build_parallel();
    let (tx, rx) = crossbeam_channel::unbounded::<(String, FileData)>();
    let writer_handle = std::thread::spawn(move || { while let Ok((p, d)) = rx.recv() { let _ = write_ndjson(&mut writer, &p, &d); } });
    walker.run(|| {
        let tx = tx.clone();
        Box::new(move |res| {
            if let Ok(entry) = res {
                let p = entry.path();
                if p.extension().map_or(false, |e| e == "rs") {
                    if let Some((rel, data)) = process_file(p, &repo_path) { let _ = tx.send((rel, data)); }
                }
            }
            ignore::WalkState::Continue
        })
    });
    drop(tx);
    let _ = writer_handle.join();
}
