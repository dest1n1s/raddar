extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn;

#[proc_macro_derive(CallableModule)]
pub fn callable_module_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;
    let gen = quote! {
        impl Fn<(&Tensor, )> for #name {
            extern "rust-call" fn call(&self, input: (&Tensor, )) -> tch::Tensor {
                self.forward(input.0)
            }
        }
        
        impl FnMut<(&Tensor, )> for #name {
            extern "rust-call" fn call_mut(&mut self, input: (&Tensor, )) -> tch::Tensor {
                self.forward(input.0)
            }
        }
        
        impl FnOnce<(&Tensor, )> for #name {
            type Output = Tensor;
        
            extern "rust-call" fn call_once(self, input: (&Tensor, )) -> Tensor {
                self.forward(input.0)
            }
        }
    };
    gen.into()
}