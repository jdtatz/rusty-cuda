#[macro_export]
macro_rules! lib_defn {
    ($feature_switch_name:literal, $link_name:literal, $lib_name:ident, { $($fname:ident : fn($($args:tt)*) -> $ret:ty),* }) => {
        #[cfg(not(feature = $feature_switch_name))]
        #[link(name=$link_name)]
        extern {
            $(
                fn $fname($($args)*) -> $ret;
            )*
        }

        #[cfg(feature = $feature_switch_name)]
        #[derive(WrapperApi)]
        struct $lib_name {
            $(
                $fname : unsafe extern "C" fn($($args)*) -> $ret,
            )*
        }
    }
}
