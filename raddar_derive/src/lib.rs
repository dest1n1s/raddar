extern crate proc_macro;

use derive_builder::Builder;
use proc_macro::TokenStream;
use quote::quote;
use syn::{self, parse::Parser, Ident, Meta, LitStr};

#[derive(Debug, Clone, Builder)]
#[builder_setter_attr(prefix = "set_")]
struct ModuleAttr {
    #[builder(default = "None", setter(strip_option))]
    pub tensor_type: Option<Ident>,

    #[builder(default = "true")]
    pub callable: bool,    

    #[builder(default = "false")]
    pub paramless: bool,

    #[builder(default = "false")]
    pub builder: bool,
}

fn check_and_get_bool_attr(meta: &Meta, name: &str) -> Option<bool> {
    match meta {
        Meta::NameValue(nv) => {
            if nv.path.is_ident(name) {
                if let syn::Lit::Bool(b) = &nv.lit {
                    return Some(b.value);
                }
            }
        },
        Meta::Path(p) => {
            if p.is_ident(name) {
                return Some(true);
            }
        },
        _ => {}
    };
    None
}

fn check_and_get_str_attr(meta: &Meta, name: &str) -> Option<LitStr> {
    match meta {
        Meta::NameValue(nv) => {
            if nv.path.is_ident(name) {
                if let syn::Lit::Str(s) = &nv.lit {
                    return Some(s.to_owned());
                }
            }
        },
        _ => {}
    };
    None
}

fn extract_module_attr(attrs: &[syn::Attribute]) -> ModuleAttr {
    let mut module_attr_builder = ModuleAttrBuilder::default();
    for attr in attrs {
        if let syn::Meta::List(list) = attr.parse_meta().unwrap() {
            if list.path.is_ident("module") {
                for nested in list.nested {
                    if let syn::NestedMeta::Meta(meta) = nested {
                        // check_and_get_bool_attr(&meta, "callable").and_then(module_attr_builder.callable);
                        if let Some(b) = check_and_get_bool_attr(&meta, "callable") {
                            module_attr_builder.callable(b);
                        }
                        if let Some(b) = check_and_get_bool_attr(&meta, "paramless") {
                            module_attr_builder.paramless(b);
                        }
                        if let Some(b) = check_and_get_bool_attr(&meta, "builder") {
                            module_attr_builder.builder(b);
                        }
                        if let Some(ident) = check_and_get_str_attr(&meta, "tensor_type") {
                            module_attr_builder.tensor_type(ident.parse().unwrap());
                        }
                    } 
                }
            }
        }
    }
    module_attr_builder.build().unwrap()
}

#[proc_macro_derive(
    Module,
    attributes(
        module,
        builder,
        builder_field_attr,
        builder_impl_attr,
        builder_setter_attr,
        builder_struct_attr
    )
)]
pub fn module_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();
    let mut output = TokenStream::new();
    let module_attr = extract_module_attr(&ast.attrs);
    output.extend(callable_derive(&ast, &module_attr));
    output.extend(paramless_derive(&ast, &module_attr));
    output.extend(architecture_builder_derive(&ast, &module_attr));



    output
}

fn callable_derive(ast: &syn::DeriveInput, attr: &ModuleAttr) -> TokenStream {
    if !attr.callable {
        return TokenStream::new();
    }

    let name = &ast.ident;

    let generics = &ast.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let tensor_type = attr.tensor_type.to_owned();

    let (impl_generics, tensor_type) = if let Some(tensor_type) = tensor_type {
        (quote! { #impl_generics }, quote! { #tensor_type })
    } else {
        let type_params = generics.type_params();
        (
            quote! { <Ts: raddar::core::TensorNN, #(#type_params),*> },
            quote! { Ts },
        )
    };

    let gen = quote! {
        impl #impl_generics Fn<(& #tensor_type, )> for #name #ty_generics #where_clause {
            extern "rust-call" fn call(&self, input: (& #tensor_type, )) -> #tensor_type {
                self.forward(input.0)
            }
        }

        impl #impl_generics FnMut<(& #tensor_type, )> for #name #ty_generics #where_clause {
            extern "rust-call" fn call_mut(&mut self, input: (& #tensor_type, )) -> #tensor_type {
                self.forward(input.0)
            }
        }

        impl #impl_generics FnOnce<(& #tensor_type, )> for #name #ty_generics #where_clause {
            type Output = #tensor_type;

            extern "rust-call" fn call_once(self, input: (& #tensor_type, )) -> #tensor_type {
                self.forward(input.0)
            }
        }

        impl #impl_generics Fn<(& #tensor_type, )> for raddar::nn::Mod<#name #ty_generics, #tensor_type> #where_clause {
            extern "rust-call" fn call(&self, input: (& #tensor_type, )) -> #tensor_type {
                self.module().forward(input.0)
            }
        }

        impl #impl_generics FnMut<(& #tensor_type, )> for raddar::nn::Mod<#name #ty_generics, #tensor_type> #where_clause {
            extern "rust-call" fn call_mut(&mut self, input: (& #tensor_type, )) -> #tensor_type {
                self.module().forward(input.0)
            }
        }

        impl #impl_generics FnOnce<(& #tensor_type, )> for raddar::nn::Mod<#name #ty_generics, #tensor_type> #where_clause {
            type Output = #tensor_type;

            extern "rust-call" fn call_once(self, input: (& #tensor_type, )) -> #tensor_type {
                self.module().forward(input.0)
            }
        }
    };
    gen.into()
}

fn paramless_derive(ast: &syn::DeriveInput, attr: &ModuleAttr) -> TokenStream {
    if !attr.paramless {
        return TokenStream::new();
    }

    let name = &ast.ident;

    let generics = &ast.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let tensor_type = attr.tensor_type.to_owned();

    let gen = if let Some(tensor_type) = tensor_type {
        quote! {
            impl #impl_generics raddar::nn::Trainable<#tensor_type> for #name #ty_generics #where_clause {}
        }
    } else {
        let type_params = generics.type_params();
        quote! {
            impl<Ts: raddar::core::TensorNN, #(#type_params),*> raddar::nn::Trainable<Ts> for #name #ty_generics #where_clause {}
        }
    };
    gen.into()
}

fn architecture_builder_derive(ast: &syn::DeriveInput, attr: &ModuleAttr) -> TokenStream {
    if !attr.builder {
        return TokenStream::new();
    }

    let data_struct = match &ast.data {
        syn::Data::Struct(data_struct) => data_struct,
        _ => panic!("ArchitectureBuilder can only be derived for structs"),
    };

    let tensor_type = attr.tensor_type.to_owned();

    let builder_fields = match data_struct.fields.to_owned() {
        syn::Fields::Named(syn::FieldsNamed { named, .. }) => {
            let mut named = named
                .into_iter()
                .filter(|field| {
                    field.attrs.iter().any(|attr| {
                        attr.path.is_ident("builder")
                            || attr.path.is_ident("builder_field_attr")
                            || attr.path.is_ident("builder_impl_attr")
                            || attr.path.is_ident("builder_setter_attr")
                            || attr.path.is_ident("builder_struct_attr")
                    })
                })
                .collect::<Vec<_>>();
            if let Some(tensor_type) = tensor_type.as_ref() {
                named.push(
                    syn::Field::parse_named
                        .parse2(quote! {
                            #[builder(default="std::marker::PhantomData")]
                            _marker_ts: std::marker::PhantomData<#tensor_type>
                        })
                        .unwrap(),
                );
            }
            named
        }
        _ => panic!("ArchitectureBuilder can only be used on structs with named fields"),
    };

    let name = &ast.ident;
    let generics = &ast.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let config_name = syn::Ident::new(&format!("{}Config", name), name.span());

    let builder_name = syn::Ident::new(&format!("{}Builder", name), name.span());
    let builder_name_str = syn::LitStr::new(&builder_name.to_string(), builder_name.span());
    let output = if let Some(tensor_type) = tensor_type.as_ref() {
        quote! {
            #[derive(derive_builder::Builder, Clone, Debug)]
            #[builder(pattern = "owned", name = #builder_name_str, build_fn(private, name = "build_config"))]
            pub struct #config_name #impl_generics #where_clause {
                #(#builder_fields),*
            }

            impl #impl_generics #builder_name #ty_generics #where_clause {
                pub fn build(self) -> raddar::nn::Mod<#name #ty_generics, #tensor_type> {
                    raddar::nn::Mod::new(#name::new(self.build_config().unwrap()))
                }
            }
        }
    } else {
        quote! {
            #[derive(derive_builder::Builder, Clone, Debug)]
            #[builder(pattern = "owned", name = #builder_name_str, build_fn(private, name = "build_config"))]
            pub struct #config_name #impl_generics #where_clause {
                #(#builder_fields),*
            }

            impl #impl_generics #builder_name #ty_generics #where_clause {
                pub fn build<Ts: raddar::core::TensorNN>(self) -> raddar::nn::Mod<#name #ty_generics, Ts> {
                    raddar::nn::Mod::new(#name::new(self.build_config().unwrap()))
                }
            }
        }
    };

    output.into()
}

#[proc_macro_derive(
    PartialBuilder,
    attributes(
        builder,
        builder_field_attr,
        builder_impl_attr,
        builder_setter_attr,
        builder_struct_attr
    )
)]
pub fn partial_builder_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input.clone()).unwrap();

    let builder_fields = match ast.data {
        syn::Data::Struct(syn::DataStruct {
            fields: syn::Fields::Named(syn::FieldsNamed { named, .. }),
            ..
        }) => named
            .into_iter()
            .filter(|field| {
                field.attrs.iter().any(|attr| {
                    attr.path.is_ident("builder")
                        || attr.path.is_ident("builder_field_attr")
                        || attr.path.is_ident("builder_impl_attr")
                        || attr.path.is_ident("builder_setter_attr")
                        || attr.path.is_ident("builder_struct_attr")
                })
            })
            .collect::<Vec<_>>(),
        _ => panic!("ArchitectureBuilder can only be used on structs with named fields"),
    };

    let name = &ast.ident;
    let generics = &ast.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let config_name = syn::Ident::new(&format!("{}Config", name), name.span());

    let builder_name = syn::Ident::new(&format!("{}Builder", name), name.span());
    let builder_name_str = syn::LitStr::new(&builder_name.to_string(), builder_name.span());
    let output = quote! {
        #[derive(derive_builder::Builder, Clone, Debug)]
        #[builder(pattern = "owned", name = #builder_name_str, build_fn(private, name = "build_config"))]
        pub struct #config_name #impl_generics #where_clause {
            #(#builder_fields),*
        }

        impl #impl_generics #builder_name #ty_generics #where_clause {
            pub fn build(self) -> #name #ty_generics {
                #name::new(self.build_config().unwrap())
            }
        }
    };
    output.into()
}

#[proc_macro_attribute]
pub fn module_state(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);

    if let syn::Fields::Named(ref mut fields) = ast.fields {
        fields.named.push(
            syn::Field::parse_named
                .parse2(quote! {
                    #[builder(default="\"a\".to_string()")]
                    pub a: String
                })
                .unwrap(),
        );
    }

    quote! {
        #ast
    }
    .into()
}

#[proc_macro_derive(DatasetIntoIter)]
pub fn dataset_into_iter_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;

    let generics = &ast.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let gen = quote! {
        impl #impl_generics IntoIterator for #name #ty_generics #where_clause {
            type Item = <Self as raddar::dataset::Dataset>::SampleType;
            type IntoIter = raddar::dataset::DatasetIterator<Self>;

            fn into_iter(self) -> Self::IntoIter {
                raddar::dataset::DatasetIterator::new(self.data())
            }
        }
    };
    gen.into()
}

#[proc_macro_derive(DatasetFromIter)]
pub fn dataset_from_iter_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;

    let generics = &ast.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let gen = quote! {
        impl #impl_generics FromIterator<<Self as raddar::dataset::Dataset>::BatchType> for #name #ty_generics #where_clause {
            fn from_iter<I: IntoIterator<Item = <Self as raddar::dataset::Dataset>::BatchType>>(iter: I) -> Self {
                Self::from_batches(iter)
            }
        }

        impl #impl_generics FromIterator<<Self as raddar::dataset::Dataset>::SampleType> for #name #ty_generics #where_clause {
            fn from_iter<I: IntoIterator<Item = <Self as raddar::dataset::Dataset>::SampleType>>(iter: I) -> Self {
                Self::from_data(iter)
            }
        }
    };
    gen.into()
}
