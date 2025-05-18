import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta

class TradingStrategy:
    def __init__(self, symbol, timeframe='1m'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = None
        
    def fetch_data(self, start_date=None, end_date=None):
        """Fetch historical data from Yahoo Finance for the specified date range"""
        ticker = yf.Ticker(self.symbol)
        
        try:
            if start_date and end_date:
                # Convert to string format that yfinance expects
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                # For intraday data, we need to handle different timeframes
                if self.timeframe in ['1m', '5m', '15m', '1h']:
                    # For intraday data, we need to fetch 1m data and resample
                    self.data = ticker.history(start=start_str, end=end_str, interval='1m', prepost=True)
                    
                    if self.data is not None and not self.data.empty:
                        # Resample to the desired timeframe
                        if self.timeframe == '5m':
                            self.data = self.data.resample('5T').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            })
                        elif self.timeframe == '15m':
                            self.data = self.data.resample('15T').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            })
                        elif self.timeframe == '1h':
                            self.data = self.data.resample('1H').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            })
                else:
                    # For daily data, fetch directly
                    self.data = ticker.history(start=start_str, end=end_str, interval=self.timeframe, prepost=True)
            else:
                # Default to last day if no dates provided
                if self.timeframe in ['1m', '5m', '15m', '1h']:
                    self.data = ticker.history(period='1d', interval='1m', prepost=True)
                    if self.data is not None and not self.data.empty:
                        # Resample to the desired timeframe
                        if self.timeframe == '5m':
                            self.data = self.data.resample('5T').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            })
                        elif self.timeframe == '15m':
                            self.data = self.data.resample('15T').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            })
                        elif self.timeframe == '1h':
                            self.data = self.data.resample('1H').agg({
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last',
                                'Volume': 'sum'
                            })
                else:
                    self.data = ticker.history(period='1d', interval=self.timeframe, prepost=True)
            
            if self.data is not None and not self.data.empty:
                # Ensure timezone information is present
                if self.data.index.tz is None:
                    self.data.index = self.data.index.tz_localize('UTC')
                return self.data
            return None
        
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None
    
    def calculate_indicators(self):
        """Calculate technical indicators"""
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
            
        # Calculate Bollinger Bands
        bb = ta.volatility.BollingerBands(close=self.data['Close'], window=20, window_dev=2)
        self.data['bb_upper'] = bb.bollinger_hband()
        self.data['bb_middle'] = bb.bollinger_mavg()
        self.data['bb_lower'] = bb.bollinger_lband()
        
        # Calculate RSI
        self.data['rsi'] = ta.momentum.rsi(self.data['Close'], window=14)
        
        return self.data
    
    def analyze_candlestick(self, index):
        """Analyze candlestick patterns"""
        if index < 1:
            return None
            
        current = self.data.iloc[index]
        previous = self.data.iloc[index-1]
        
        # Basic candlestick patterns
        body_size = abs(current['Close'] - current['Open'])
        upper_shadow = current['High'] - max(current['Open'], current['Close'])
        lower_shadow = min(current['Open'], current['Close']) - current['Low']
        
        # Bullish pattern
        if (current['Close'] > current['Open'] and  # Bullish candle
            body_size > 0.5 * (current['High'] - current['Low']) and  # Strong body
            lower_shadow < body_size * 0.3):  # Small lower shadow
            return 'bullish'
            
        # Bearish pattern
        elif (current['Close'] < current['Open'] and  # Bearish candle
              body_size > 0.5 * (current['High'] - current['Low']) and  # Strong body
              upper_shadow < body_size * 0.3):  # Small upper shadow
            return 'bearish'
            
        return None
    
    def check_bollinger_breakout(self, index):
        """Check for Bollinger Band breakouts and positions beyond bands"""
        if index < 1:
            return None
            
        current = self.data.iloc[index]
        previous = self.data.iloc[index-1]
        
        # Calculate how far price is from the bands (as a percentage)
        upper_distance = ((current['Close'] - current['bb_upper']) / current['bb_upper']) * 100
        lower_distance = ((current['bb_lower'] - current['Close']) / current['bb_lower']) * 100
        
        # Upper band breakout or position
        if current['Close'] > current['bb_upper']:
            # If it's a crossover or significantly above the band
            if (previous['Close'] <= previous['bb_upper'] or upper_distance > 0.1):
                return 'upper_breakout'
            
        # Lower band breakout or position
        elif current['Close'] < current['bb_lower']:
            # If it's a crossover or significantly below the band
            if (previous['Close'] >= previous['bb_lower'] or lower_distance > 0.1):
                return 'lower_breakout'
            
        return None
    
    def check_rsi_extremes(self, index):
        """Check for RSI extreme values"""
        if index < 1:
            return None
            
        current = self.data.iloc[index]
        
        if current['rsi'] > 70:
            return 'overbought'
        elif current['rsi'] < 30:
            return 'oversold'
            
        return None
    
    def generate_signals(self):
        """Generate trading signals and indicator values for every minute"""
        signals = []
        last_buy_signal = None  # Track the last buy signal
        
        for i in range(1, len(self.data)):
            signal = {
                'timestamp': self.data.index[i],
                'price': self.data['Close'].iloc[i],
                'bb_upper': self.data['bb_upper'].iloc[i],
                'bb_middle': self.data['bb_middle'].iloc[i],
                'bb_lower': self.data['bb_lower'].iloc[i],
                'rsi': self.data['rsi'].iloc[i],
                'candlestick': self.analyze_candlestick(i),
                'action': None,
                'reason': []
            }
            
            # Get current and previous candlestick data
            current_high = self.data['High'].iloc[i]
            current_low = self.data['Low'].iloc[i]
            current_close = self.data['Close'].iloc[i]
            current_open = self.data['Open'].iloc[i]
            
            previous_high = self.data['High'].iloc[i-1]
            previous_low = self.data['Low'].iloc[i-1]
            previous_close = self.data['Close'].iloc[i-1]
            previous_open = self.data['Open'].iloc[i-1]
            
            # Get current and previous BB levels
            current_bb_upper = self.data['bb_upper'].iloc[i]
            current_bb_lower = self.data['bb_lower'].iloc[i]
            current_bb_middle = self.data['bb_middle'].iloc[i]
            previous_bb_upper = self.data['bb_upper'].iloc[i-1]
            previous_bb_lower = self.data['bb_lower'].iloc[i-1]
            
            # Determine BB position based on candlestick
            if current_low > current_bb_upper:
                # Candlestick is completely above upper BB
                signal['bb_position'] = 'Above Upper BB'
            elif current_high < current_bb_lower:
                # Candlestick is completely below lower BB
                signal['bb_position'] = 'Below Lower BB'
            elif current_low <= current_bb_upper and current_high >= current_bb_upper:
                # Candlestick is crossing upper BB
                if previous_high < previous_bb_upper:
                    signal['bb_position'] = 'Crossing Above Upper BB'
                else:
                    signal['bb_position'] = 'Touching Upper BB'
            elif current_low <= current_bb_lower and current_high >= current_bb_lower:
                # Candlestick is crossing lower BB
                if previous_low > previous_bb_lower:
                    signal['bb_position'] = 'Crossing Below Lower BB'
                else:
                    signal['bb_position'] = 'Touching Lower BB'
            else:
                # Candlestick is between the bands
                signal['bb_position'] = 'Inside BB'
            
            # Check Bollinger Band breakout
            bb_breakout = self.check_bollinger_breakout(i)
            if bb_breakout:
                signal['reason'].append(f'Bollinger: {bb_breakout}')
            
            # Check RSI extremes
            rsi_signal = self.check_rsi_extremes(i)
            if rsi_signal:
                signal['reason'].append(f'RSI: {rsi_signal}')
            
            # Generate trading action based on signals
            if bb_breakout == 'upper_breakout' and rsi_signal == 'overbought':
                signal['action'] = 'SELL'
            elif (bb_breakout == 'lower_breakout' and rsi_signal == 'oversold') or \
                 (current_low < current_bb_lower and rsi_signal == 'oversold' and current_close < previous_close):
                # Allow consecutive buy signals only if price is making new lows
                if last_buy_signal is None or current_close < last_buy_signal['price']:
                    signal['action'] = 'BUY'
                    last_buy_signal = signal  # Store the buy signal
                    
                    # Add more detailed reasons
                    if bb_breakout == 'lower_breakout':
                        signal['reason'].append('Bollinger: lower_breakout')
                    if current_low < current_bb_lower:
                        signal['reason'].append('Price below lower BB')
                    if rsi_signal == 'oversold':
                        signal['reason'].append('RSI: oversold')
            
            # Check for sell signal after a buy
            if last_buy_signal is not None:
                # Calculate 50% point between middle and upper BB
                fifty_percent_point = current_bb_middle + (current_bb_upper - current_bb_middle) * 0.75
                
                if current_high >= current_bb_upper:
                    signal['action'] = 'SELL'
                    signal['reason'].append('Take profit: Upper BB cross after buy')
                    last_buy_signal = None  # Reset the buy signal
                elif current_high >= fifty_percent_point:
                    signal['action'] = 'SELL'
                    signal['reason'].append('Take profit: Reached 75% between middle and upper BB')
                    last_buy_signal = None  # Reset the buy signal
            
            signals.append(signal)
        
        return signals

def main():
    # Example usage
    strategy = TradingStrategy('^GSPC')  # S&P 500
    data = strategy.fetch_data()
    strategy.calculate_indicators()
    signals = strategy.generate_signals()
    
    # Print signals
    for signal in signals:
        print(f"Time: {signal['timestamp']}")
        print(f"Price: {signal['price']}")
        print(f"Action: {signal['action']}")
        print(f"Reasons: {', '.join(signal['reason'])}")
        print("-" * 50)

if __name__ == "__main__":
    main()
