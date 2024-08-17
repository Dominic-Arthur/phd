import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)


def plot_epi_time_scales_model(R0=1.5, latent_period=4.5, incubation_period=5.6, num_samples=1000):
    # Function to update the plots based on parameter changes
    def update_plots(lp, ip, scale_inf_trans, scale_sym_test, noise):
        inf_sym_mean = ip - lp
        if inf_sym_mean < 0:
            st.error("Latent Period should be less than the Incubation Period")
        else:
            scale_inf_trans += noise
            scale_sym_test += noise

            exp_inf = rng.lognormal(mean=np.log(lp), sigma=np.log(1.5), size=num_samples)
            inf_sym = exp_inf + rng.lognormal(mean=np.log(inf_sym_mean), sigma=np.log(1), size=num_samples)
            inf_trans = exp_inf + rng.gamma(shape=R0, scale=scale_inf_trans, size=num_samples)
            sym_test = rng.gamma(shape=1, scale=scale_sym_test, size=num_samples)
            inf_trest = inf_sym + sym_test

            trans_within_range = (inf_trans >= inf_sym) & (inf_trans <= inf_sym + 5)
            prop_within_range = np.mean(trans_within_range) * 100
            prop_trans_before_testing = np.mean((inf_trest >= inf_trans)) * 100
            prop_pre_sym_cases = np.mean((inf_trans <= inf_sym)) * 100

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
            plt.title("Epidemic Time Scale Dynamics of Symptomatic Infections")
            plt.legend(loc='center right')
            st.pyplot(plt)

            st.subheader("Summary Statistics")
            st.write(f"**Average generation time**: {inf_trans.mean():.1f} days")
            st.write(f"**{prop_within_range:.2f}%** transmissions occurred within 5 days after onset of symptoms")
            st.write(f"**{prop_trans_before_testing:.2f}%** transmissions occurred before testing")
            st.write(f"**{prop_pre_sym_cases:.2f}%** pre-symptomatic transmissions")

    # Streamlit sliders
    latent_periods = st.sidebar.slider("Latent Period", 1.0, 10.0, latent_period, 0.1,
                                       help="The time period between exposure and becoming infectious")
    incubation_periods = st.sidebar.slider("Incubation Period", 1.1, 15.0, incubation_period, 0.1,
                                           help="The time period between exposure and symptom onset")

    scale_inf_trans_values = st.sidebar.slider("Scale - Inf_Trans", 0.0, 10.0, 0.0, 0.1,
                                               help="Amount of variability in the infectious-to-transmission period")
    scale_sym_test_values = st.sidebar.slider("Scale - Sym_Test", 0.0, 10.0, 0.0, 0.1,
                                              help="Amount of variability in the symptom-to-testing period")
    noise_values = st.sidebar.slider("Alpha", 0.0, 10.0, 1.0, 0.1,
                                     help="Adjust variability, scaled be the Beta parameters")

    update_plots(latent_periods, incubation_periods, scale_inf_trans_values, scale_sym_test_values, noise_values)


if __name__ == "__main__":
    st.header("Modelling Epidemic Time Scale Dynamics of Symptomatic Infections")
    st.write(
        "Use the sliders in the sidebar to adjust the parameters and observe the changes in the epidemic dynamics.")

    plot_epi_time_scales_model()
