

month 07 produces NaN values because data from 03-07-2023 after 12:45 pm is not available. For now skip it.
ask price is not original ask price. Must keep track of original "ask" and "bid" cols for coherent money evaluation. Has to be modified.
Optional -> if problem persist must increase q_low and increase q_high for more strict filtering.

There is a bug in the code. When I load the source already saved, the code breaks on:
`print(f"all labels match: {all([lab_batch == df_train_copy['label'].iloc[idx] for idx, lab_batch in enumerate(train_labels)])}")`

I have to do extensive data exploration. Modeling is not working so far