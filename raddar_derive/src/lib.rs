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