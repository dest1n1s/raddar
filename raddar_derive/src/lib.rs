extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn;

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
    let config_name = syn::Ident::new(&format!("{}Config", name), name.span());

    let builder_name = syn::Ident::new(&format!("{}Builder", name), name.span());
    let builder_name_str = syn::LitStr::new(&builder_name.to_string(), builder_name.span());
    let output = quote! {
        #[derive(derive_builder::Builder, Clone, Debug)]
        #[builder(pattern = "owned", name = #builder_name_str, build_fn(private, name = "build_config"))]
        pub struct #config_name {
            #(#builder_fields),*
        }

        impl #builder_name {
            pub fn build(self) -> #name {
                #name::new(self.build_config().unwrap())
            }
        }
    };
    output.into()
}

#[proc_macro_derive(IterableDataset)]
pub fn iterable_dataset_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;

    let generics = &ast.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let gen = quote! {
        impl #impl_generics IntoIterator for #name #ty_generics #where_clause {
            type Item = <Self as raddar::dataset::Dataset>::BatchType;
            type IntoIter = raddar::dataset::DatasetIterator<Self>;

            fn into_iter(self) -> Self::IntoIter {
                let batch_size = self.batch_size();
                Self::IntoIter::new(self.data(), batch_size)
            }
        }
    };
    gen.into()
}