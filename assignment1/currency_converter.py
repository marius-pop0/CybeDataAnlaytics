class CurrencyConverter:

    RATES_AUD = {
        'AUD': 1.0,
        'SEK': 0.151,
        'MXN': 0.069,
        'GBP': 1.795,
        'NZD': 0.93
    }

    ### df[['currencycode', 'amount']]
    @staticmethod
    def convert_currency_from_AUD(x):
        return CurrencyConverter.RATES_AUD[x[0]] * x[1]