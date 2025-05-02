import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV data
data = pd.read_csv('results.csv')
sizes = data['Size']
Single_Thread = data['Single_Thread']
OpenMp = data['OpenMp']
OpenMpWithThreads = data['OpenMpWithThreads']

# Create the plot
plt.figure(figsize=(12, 7))
plt.plot(sizes, Single_Thread, 'b-o', label='Single_Thread')
plt.plot(sizes, OpenMp, 'g-^', label='OpenMp')
plt.plot(sizes, OpenMpWithThreads, 'r-s', label='OpenMpWithThreads')

plt.xlabel('Input Size (log scale)')
plt.ylabel('Execution Time (ms, log scale)')
plt.title('FFT Multiplication Performance vs Input Size')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

# Add annotations for the last data points
for x, y, label in [(sizes.iloc[-1], Single_Thread.iloc[-1], 'Single_Thread'),
                   (sizes.iloc[-1], OpenMp.iloc[-1], 'OpenMp'),
                   (sizes.iloc[-1], OpenMpWithThreads.iloc[-1], 'OpenMpWithThreads')]:
    plt.annotate(f'', (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center')

plt.savefig('fft_performance.png', dpi=300)
plt.show()