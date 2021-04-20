_py = pd.DataFrame({
    'Close': [1,e,e**2],
    'Volume': [1,2,1]
})
_thon = pd.DataFrame({
    'Close': [e**2,e,1],
    'Volume': [1,4,10]
})
_test_df = pd.concat([_py,_thon], axis=1, keys=["Py", "Thon"])
_test_df.columns.set_names(["asset", "value"], inplace=True)

close = _test_df.xs('Close', level='value', axis=1)
volume = _test_df.xs('Volume', level='value', axis=1)
_test_lrInd = LR.run(close)