extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{self, parse::Parser};

#[proc_macro_derive(CallableModule)]
pub fn callable_module_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;

    let generics = &ast.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let gen = quote! {
        impl #impl_generics Fn<(&Tensor, )> for #name #ty_generics #where_clause {
            extern "rust-call" fn call(&self, input: (&Tensor, )) -> tch::Tensor {
                self.forward(input.0)
            }
        }

        impl #impl_generics FnMut<(&Tensor, )> for #name #ty_generics #where_clause {
            extern "rust-call" fn call_mut(&mut self, input: (&Tensor, )) -> tch::Tensor {
                self.forward(input.0)
            }
        }

        impl #impl_generics FnOnce<(&Tensor, )> for #name #ty_generics #where_clause {
            type Output = Tensor;

            extern "rust-call" fn call_once(self, input: (&Tensor, )) -> Tensor {
                self.forward(input.0)
            }
        }
    };
    gen.into()
}

#[proc_macro_derive(NonParameterModule)]
pub fn non_parameter_module_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;

    let generics = &ast.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let gen = quote! {
        impl #impl_generics raddar::nn::NonParameterModule for #name #ty_generics #where_clause {}
    };
    gen.into()
}

#[proc_macro_derive(
    ArchitectureBuilder,
    attributes(
        builder,
        builder_field_attr,
        builder_impl_attr,
        builder_setter_attr,
        builder_struct_attr
    )
)]
pub fn architecture_builder_derive(input: TokenStream) -> TokenStream {
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
            pub fn build(self) -> raddar::nn::Mod<#name #ty_generics> {
                raddar::nn::Mod::new(#name::new(self.build_config().unwrap()))
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
