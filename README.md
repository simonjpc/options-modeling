# options-modeling
experiments for market options value modeling


With z_score > 3 still drop in month 10 and 12
With quantiles I use 0.01 and 0.99, not quite equivalent to z = 3 (0.0013, 0.9987)


sub_df_quantiles -> 1000 to 14
sub_df_high_entropy -> 1000 to 
sub_df_ff_centroid -> 1000 to 
sub_df_low_similarity -> 1000 to 
sub_df_inliers -> 1000 to 

--

Check calculation for the following features that provide a great number of null values:

- beta,
- bear_beta,
- std_amihud

Check calculation for the following features that provide some null values:

- ailliq
- amihudilliq
- norm_mom_1
- norm_mom_2
- norm_mom_3
- norm_mom_4
- norm_mom_5
- norm_mom_10
- norm_mom_15
- psliq
- historic_kurtosis
- historic_skewness
- historic_volatility
- iv_minus_realized
- iv_minus_realized_ratio
- return
- pfht
- pifht
- piroll
- retvol
- std_dolvol
- scaled_volga
- zerotrade
