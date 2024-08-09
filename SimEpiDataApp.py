import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)


def plot_epi_time_scales_model(latent_period=4.0, incubation_period=5.0, betas=None, num_samples=1000):
    if betas is None:
        betas = {"noise_exp_inf": 5.0, "noise_inf_sym": 5.0, "scale_inf_trans": 2.0, "scale_sym_test": 1.0}

    def update_plots(lp, ip, noise_exp_inf, noise_inf_sym, scale_inf_trans, scale_sym_test,
                     alpha):
        inf_sym_mean = ip - lp
        if inf_sym_mean < 0:
            st.error("Latent Period should be less than the Incubation Period")
        else:
            noise_exp_inf += alpha / betas["noise_exp_inf"]
            noise_inf_sym += alpha / betas["noise_inf_sym"]
            scale_inf_trans += alpha / betas["scale_inf_trans"]
            scale_sym_test += alpha / betas["scale_sym_test"]

            exp_inf = rng.lognormal(mean=np.log(lp), sigma=np.log(noise_exp_inf), size=num_samples)
            inf_sym = exp_inf + rng.lognormal(mean=np.log(inf_sym_mean), sigma=np.log(noise_inf_sym), size=num_samples)
            inf_trans = exp_inf + rng.gamma(shape=4, scale=scale_inf_trans, size=num_samples)
            sym_test = rng.gamma(shape=3, scale=scale_sym_test, size=num_samples)
            inf_trest = inf_sym + sym_test

            trans_within_range = (inf_trans >= inf_sym) & (inf_trans <= inf_sym + 5)
            prop_within_range = np.mean(trans_within_range) * 100
            prop_trans_before_testing = np.mean((inf_trest >= inf_trans)) * 100
            prop_pre_sym_cases = np.mean((inf_trans <= inf_sym)) * 100

            st.write(f"**Average generation time**: {inf_trans.mean():.1f} days")
            st.write(f"**{prop_within_range:.2f}%** transmissions occurred within 5 days after onset of symptoms")
            st.write(f"**{prop_trans_before_testing:.2f}%** transmissions occurred before testing")
            st.write(f"**{prop_pre_sym_cases:.2f}%** pre-symptomatic transmissions")

            sns.set_style('white')
            custom_params = {"axes.spines.right": False, "axes.spines.top": False}
            sns.set_theme(style="ticks", rc=custom_params)
            sns.set_context("paper", font_scale=2)

            plt.figure(figsize=(10, 6))
            sns.kdeplot(exp_inf, color='orange', fill=True, label='Infectiousness')
            sns.kdeplot(inf_trans, color='green', fill=True, label='Transmissions')
            sns.kdeplot(inf_sym, color='blue', fill=True, label='Symptom Onset')
            sns.kdeplot(inf_trest, color='red', fill=True, label='Testing')
            plt.xlabel('Time From Exposure (Days)')
            plt.ylabel('Density')
            plt.title("Epidemic Time Scale Dynamics")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
            st.pyplot(plt)

    # Streamlit sliders
    latent_periods = st.sidebar.slider("Latent Period", 1.0, 10.0, latent_period, 0.1)
    incubation_periods = st.sidebar.slider("Incubation Period", 1.1, 15.0, incubation_period, 0.1)
    noise_exp_inf_values = st.sidebar.slider("Noise Exp_Inf", 1.01, 2.0, 1.01, 0.01)
    noise_inf_sym_values = st.sidebar.slider("Noise Inf_Sym", 1.01, 2.0, 1.01, 0.01)
    scale_inf_trans_values = st.sidebar.slider("Scale Inf_Trans", 0.0, 2.0, 0.0, 0.01)
    scale_sym_test_values = st.sidebar.slider("Scale Sym_Test", 0.0, 2.0, 0.0, 0.01)
    alpha_values = st.sidebar.slider("Alpha", 0.0, 2.0, 0.5, 0.01)

    update_plots(latent_periods, incubation_periods, noise_exp_inf_values,
                 noise_inf_sym_values, scale_inf_trans_values, scale_sym_test_values, alpha_values)


if __name__ == "__main__":
    st.title("Modelling Epidemic Time Scale Dynamics of Symptomatic Infections")
    st.write(
        "Use the sliders in the sidebar to adjust the parameters and observe the changes in the epidemic dynamics.")

    st.header("Epidemiological Data 📊📊📊📊📊")
    plot_epi_time_scales_model()
