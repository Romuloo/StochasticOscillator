import ccxt
from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)


class CryptoAnalyzer:
    """
    Clase para analizar datos de criptomonedas y generar gráficos.
    """

    def __init__(self):
        """
        Constructor de la clase CryptoAnalyzer.
        """
        self.k = ccxt.kraken()

    def download_data(self, pair: str, interval: str = '1d', limit: int = 100) -> pd.DataFrame:
        """
        Descarga datos de velas desde la API de Kraken.

        :param pair: Par de monedas (por ejemplo, 'BTC/USD').
        :param interval: Intervalo de tiempo para los datos de velas (predeterminado: '1d').
        :param limit: Número de velas a descargar (predeterminado: 100).
        :return: DataFrame con los datos descargados.
        :raise ValueError: Si hay un error durante la descarga.
        """
        try:
            ohlc_data = self.k.fetch_ohlcv(pair, interval, limit=limit)
            df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except ccxt.NetworkError:
            raise ValueError("Error de red. Por favor, verifica tu conexión a Internet.")
        except ccxt.ExchangeError as e:
            error_message = f"Error al obtener datos: {e}. Consulta la documentación para obtener ayuda."
            error_message += f" <a href='https://api.kraken.com/0/public/AssetPairs' target='_blank'>Lista de pares de monedas</a>"
            raise ValueError(error_message)

    def find_divergences(self, ax, df: pd.DataFrame) -> None:
        """
        Busca divergencias en el oscilador estocástico y grafica señales.

        :param ax: Subgráfica para la visualización.
        :param df: DataFrame con los datos.
        """
        bearish_divergence = (df['close'] > df['close'].shift(1)) & (df['sd'] < df['sd'].shift(1))
        bullish_divergence = (df['close'] < df['close'].shift(1)) & (df['sd'] > df['sd'].shift(1))

        ax.axhline(80, color='red', linestyle='--', label='Overbought (80)')
        ax.axhline(20, color='green', linestyle='--', label='Oversold (20)')

        ax.plot(df.index[bearish_divergence], df['sd'][bearish_divergence], 'ro', label='Bearish Divergence')
        ax.plot(df.index[bullish_divergence], df['sd'][bullish_divergence], 'go', label='Bullish Divergence')

        ax.legend()

    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, sk_period: int = 3,
                             sd_period: int = 3) -> None:
        """
        Calcula el oscilador estocástico y agrega columnas al DataFrame.

        :param df: DataFrame con los datos.
        :param k_period: Período para %K (predeterminado: 14).
        :param sk_period: Período para %K suavizado (predeterminado: 3).
        :param sd_period: Período para %D suavizado (predeterminado: 3).
        """
        df['k'] = (df['close'] - df['low'].rolling(window=k_period).min()) / (
                df['high'].rolling(window=k_period).max() - df['low'].rolling(window=k_period).min()) * 100
        df['sk'] = df['k'].rolling(window=sk_period).mean()
        df['sd'] = df['sk'].rolling(window=sd_period).mean()

    def plot_prices_and_oscillators_base64(self, df: pd.DataFrame, pair: str) -> str:
        """
        Grafica las cotizaciones y el oscilador estocástico y devuelve la representación base64 de la imagen.

        :param df: DataFrame con los datos.
        :param pair: Par de monedas.
        :return: Representación base64 de la imagen.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(df.index, df['close'], label=f'{pair} Close Price', color='blue')
        ax1.set_ylabel('Close Price (USD)', color='blue')
        ax1.tick_params('y', colors='blue')

        self.calculate_stochastic(df)
        ax2.plot(df.index, df['sk'], label='Stochastic Oscillator %K', color='orange')
        ax2.plot(df.index, df['sd'], label='Stochastic Oscillator %D', color='green')
        ax2.set_ylabel('Value', color='black')
        ax2.tick_params('y', colors='black')

        plt.title(f'{pair} Close Price and Stochastic Oscillator Over Time')
        ax2.set_xlabel('Time')
        ax1.legend()
        ax2.legend()

        self.find_divergences(ax2, df)

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()

        return plot_base64


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Función para manejar las solicitudes HTTP en la ruta principal ('/').
    """
    crypto_analyzer = CryptoAnalyzer()

    if request.method == 'POST':
        try:
            pair = request.form['pair']
            limit = int(request.form['limit'])

            df = crypto_analyzer.download_data(pair, limit=limit)

            plot_base64 = crypto_analyzer.plot_prices_and_oscillators_base64(df, pair)

            return render_template('index.html', plot_base64=plot_base64)

        except ValueError as e:
            error_message = str(e)

    else:
        error_message = None

    return render_template('index.html', error_message=error_message, plot_base64=None)


if __name__ == '__main__':
    app.run(debug=True)
