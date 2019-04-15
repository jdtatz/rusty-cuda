#[macro_export]
macro_rules! lib_defn {
    ($link_name:literal, $lib_name:ident, { $($fname:ident : fn($($args:tt)*) -> $ret:ty),* }) => {
        #[cfg(not(feature = "dynamic"))]
        #[link(name=$link_name)]
        extern {
            $(
                fn $fname($($args)*) -> $ret;
            )*
        }

        #[cfg(feature = "dynamic")]
        #[derive(WrapperApi)]
        struct $lib_name {
            $(
                $fname : unsafe extern "C" fn($($args)*) -> $ret,
            )*
        }
    }
}
