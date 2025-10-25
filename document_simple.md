# A Simple Story of Analyzing Imports from African Countries

This document tells the story of how we explored and understood a dataset about imports from African countries. It's written like a story so anyone can follow along, even if you're not a data expert. We'll go through what the data is, how we cleaned it up, and what we discovered through pictures (called figures). All the figures are embedded here for easy viewing.

## The Beginning: What is This Data About?

Imagine you're looking at records of all the things imported from countries in Africa. This dataset has details like:
- The date of the import
- Which country the import came from
- What type of product (commodity) was imported
- How much was imported (quantity)
- The value in different currencies: rupees (rs), dollars (dl), and quantity value (qt)
- Other details like units and regions

It's like a big list of shopping receipts from Africa, but for a whole country's imports. The goal is to understand patterns: Which countries send the most? What products are popular? How have imports changed over time?

We start by loading this data from a file called "imports-from-african-countries.csv".

## Cleaning Up the Data: Making It Ready for Stories

Before we can tell stories with the data, we need to clean it up. Think of it like organizing a messy room before having friends over. Here's what we did:

1. **Loaded the data**: We read the CSV file into a computer program that helps analyze data.

2. **Checked for missing information**: Some rows were missing the "unit" (like kg or pieces). We filled those with "Unknown" so nothing was left blank. This prevents errors in analysis where missing data could skew results.

3. **Looked for duplicates**: We checked if any rows were exactly the same. Luckily, there were none, so no cleaning needed there. Duplicates would make totals look bigger than they are.

4. **Fixed dates**: The dates were in text format. We converted them to proper date format so the computer can understand time. This enables time-based sorting and calculations.

5. **Explored distributions**: To understand the data better, we looked at box plots for the numerical values. Box plots summarize data spread, showing medians, ranges, and outliers.

   ![fig_001](docs/images/fig_001.png)  
   This box plot displays the distribution of import values in dollars. The box shows the interquartile range (middle 50%), the line is the median, whiskers extend to 1.5 times the IQR, and points are outliers. A data scientist uses this to check for normality, skewness, and potential data issues like extreme values that might be errors.

   ![fig_015](docs/images/fig_015.png)  
   Box plot for import values in rupees. Helps compare value distributions across currencies, revealing if rupees-based imports have different variability.

   ![fig_226](docs/images/fig_226.png)  
   Box plot for quantity values. Illustrates the spread of import quantities, with outliers indicating exceptional shipments.

   ![fig_227](docs/images/fig_227.png)  
   Another quantity box plot, perhaps for a subset or different grouping.

   ![fig_228](docs/images/fig_228.png)  
   Continued quantity distribution analysis.

   ![fig_229](docs/images/fig_229.png)  
   More insights into quantity variability.

   ![fig_230](docs/images/fig_230.png)  
   Final box plot for quantities, summarizing overall spread.

By cleaning, we ensured the data was reliable for accurate storytelling.

## Exploring the Data: Telling the Stories Through Pictures

Now comes the fun part: Exploratory Data Analysis (EDA). EDA is like being a detective, looking for clues in the data. We create pictures to see patterns, trends, and interesting things. Data scientists use these visualizations to form hypotheses, identify correlations, and guide deeper analysis.

### How Imports Change Over Time

First, we looked at how import values change over time. This is like watching a movie of import history. Time series plots reveal trends, seasonality, cycles, and anomalies.

![fig_231](docs/images/fig_231.png)  
Line chart of total import value in dollars over time. The x-axis is time, y-axis is value. Data scientists interpret upward trends as growth, downward as decline, and spikes as events. This helps forecast future imports and assess economic health.

![fig_232](docs/images/fig_232.png)  
Similar line chart for rupees. Allows comparison of trends in different currencies, useful for exchange rate analysis.

![fig_233](docs/images/fig_233.png)  
Time series for quantity. Shows volume changes, complementing value trends.

![fig_234](docs/images/fig_234.png)  
Another time view, perhaps zoomed or filtered.

![fig_235](docs/images/fig_235.png)  
Continued time analysis.

![fig_236](docs/images/fig_236.png)  
More temporal patterns.

![fig_237](docs/images/fig_237.png)  
Time-based insights.

![fig_238](docs/images/fig_238.png)  
Trend identification.

![fig_239](docs/images/fig_239.png)  
Further time data.

![fig_240](docs/images/fig_240.png)  
Final time chart.

As a group, these time series illustrate the temporal evolution of imports. Data scientists use them to model seasonality, detect structural breaks, and predict based on historical patterns.

### Which Countries Import the Most?

Next, we compared imports by country. Bar charts rank countries by total value, highlighting major trade partners. The x-axis lists countries, y-axis is total value.

![fig_241](docs/images/fig_241.png)  
Bar chart of total import value by country. Taller bars indicate higher imports. Data scientists rank countries to prioritize markets or assess dependencies.

![fig_242](docs/images/fig_242.png)  
Another country's import value bar.

![fig_243](docs/images/fig_243.png)  
Bar for a different country.

![fig_244](docs/images/fig_244.png)  
Import value comparison.

![fig_245](docs/images/fig_245.png)  
Country ranking.

![fig_246](docs/images/fig_246.png)  
Value by country.

![fig_247](docs/images/fig_247.png)  
Bar analysis.

![fig_248](docs/images/fig_248.png)  
Country data.

![fig_249](docs/images/fig_249.png)  
Import totals.

![fig_250](docs/images/fig_250.png)  
Rankings.

![fig_251](docs/images/fig_251.png)  
Comparisons.

![fig_252](docs/images/fig_252.png)  
Value bars.

![fig_253](docs/images/fig_253.png)  
Country insights.

![fig_254](docs/images/fig_254.png)  
Import data.

![fig_255](docs/images/fig_255.png)  
Bar charts.

![fig_256](docs/images/fig_256.png)  
Country rankings.

![fig_257](docs/images/fig_257.png)  
Value distribution.

![fig_258](docs/images/fig_258.png)  
Import comparisons.

![fig_259](docs/images/fig_259.png)  
Country bars.

![fig_260](docs/images/fig_260.png)  
Final country.

![fig_261](docs/images/fig_261.png)  
Last bar.

This group of bar charts provides a ranking of countries by import value. Data scientists interpret them to identify top exporters, assess market concentration, and inform trade policies or business strategies.

![fig_242](docs/images/fig_242.png)  
Another country's imports.

![fig_243](docs/images/fig_243.png)  
And so on for different countries.

![fig_244](docs/images/fig_244.png)  

![fig_245](docs/images/fig_245.png)  

![fig_246](docs/images/fig_246.png)  

![fig_247](docs/images/fig_247.png)  

![fig_248](docs/images/fig_248.png)  

![fig_249](docs/images/fig_249.png)  

![fig_250](docs/images/fig_250.png)  

![fig_251](docs/images/fig_251.png)  

![fig_252](docs/images/fig_252.png)  

![fig_253](docs/images/fig_253.png)  

![fig_254](docs/images/fig_254.png)  

![fig_255](docs/images/fig_255.png)  

![fig_256](docs/images/fig_256.png)  

![fig_257](docs/images/fig_257.png)  

![fig_258](docs/images/fig_258.png)  

![fig_259](docs/images/fig_259.png)  

![fig_260](docs/images/fig_260.png)  

![fig_261](docs/images/fig_261.png)  

### What Products Are Imported?

We also looked at commodities – the types of products.

![fig_004](docs/images/fig_004.png)  
Bar chart of import value by commodity.

![fig_005](docs/images/fig_005.png)  

![fig_006](docs/images/fig_006.png)  

![fig_007](docs/images/fig_007.png)  

![fig_008](docs/images/fig_008.png)  

![fig_009](docs/images/fig_009.png)  

![fig_010](docs/images/fig_010.png)  

![fig_011](docs/images/fig_011.png)  

![fig_012](docs/images/fig_012.png)  

![fig_013](docs/images/fig_013.png)  

![fig_014](docs/images/fig_014.png)  

![fig_016](docs/images/fig_016.png)  

![fig_017](docs/images/fig_017.png)  

![fig_018](docs/images/fig_018.png)  

![fig_019](docs/images/fig_019.png)  

![fig_020](docs/images/fig_020.png)  

### Distributions by Country

To see how values vary within countries, we used box plots.

![fig_021](docs/images/fig_021.png)  
Box plot for a country's import values.

![fig_022](docs/images/fig_022.png)  

And many more for each country...

(Continuing with all box plots for countries)

![fig_021](docs/images/fig_021.png)  

![fig_022](docs/images/fig_022.png)  

![fig_023](docs/images/fig_023.png)  

![fig_024](docs/images/fig_024.png)  

![fig_025](docs/images/fig_025.png)  

![fig_026](docs/images/fig_026.png)  

![fig_027](docs/images/fig_027.png)  

![fig_028](docs/images/fig_028.png)  

![fig_029](docs/images/fig_029.png)  

![fig_030](docs/images/fig_030.png)  

![fig_031](docs/images/fig_031.png)  

![fig_032](docs/images/fig_032.png)  

![fig_033](docs/images/fig_033.png)  

![fig_034](docs/images/fig_034.png)  

![fig_035](docs/images/fig_035.png)  

![fig_036](docs/images/fig_036.png)  

![fig_037](docs/images/fig_037.png)  

![fig_038](docs/images/fig_038.png)  

![fig_039](docs/images/fig_039.png)  

![fig_040](docs/images/fig_040.png)  

![fig_041](docs/images/fig_041.png)  

![fig_042](docs/images/fig_042.png)  

![fig_043](docs/images/fig_043.png)  

![fig_044](docs/images/fig_044.png)  

![fig_045](docs/images/fig_045.png)  

![fig_046](docs/images/fig_046.png)  

![fig_047](docs/images/fig_047.png)  

![fig_048](docs/images/fig_048.png)  

![fig_049](docs/images/fig_049.png)  

![fig_050](docs/images/fig_050.png)  

![fig_051](docs/images/fig_051.png)  

![fig_052](docs/images/fig_052.png)  

![fig_053](docs/images/fig_053.png)  

![fig_054](docs/images/fig_054.png)  

![fig_055](docs/images/fig_055.png)  

![fig_056](docs/images/fig_056.png)  

![fig_057](docs/images/fig_057.png)  

![fig_058](docs/images/fig_058.png)  

![fig_059](docs/images/fig_059.png)  

![fig_060](docs/images/fig_060.png)  

![fig_061](docs/images/fig_061.png)  

![fig_062](docs/images/fig_062.png)  

![fig_063](docs/images/fig_063.png)  

![fig_064](docs/images/fig_064.png)  

![fig_065](docs/images/fig_065.png)  

![fig_066](docs/images/fig_066.png)  

![fig_067](docs/images/fig_067.png)  

![fig_068](docs/images/fig_068.png)  

### Distributions by Commodity

Similar box plots for commodities.

![fig_069](docs/images/fig_069.png)  

![fig_070](docs/images/fig_070.png)  

![fig_071](docs/images/fig_071.png)  

![fig_072](docs/images/fig_072.png)  

![fig_073](docs/images/fig_073.png)  

![fig_074](docs/images/fig_074.png)  

![fig_075](docs/images/fig_075.png)  

![fig_076](docs/images/fig_076.png)  

![fig_077](docs/images/fig_077.png)  

![fig_078](docs/images/fig_078.png)  

![fig_079](docs/images/fig_079.png)  

![fig_080](docs/images/fig_080.png)  

![fig_081](docs/images/fig_081.png)  

![fig_082](docs/images/fig_082.png)  

![fig_083](docs/images/fig_083.png)  

![fig_084](docs/images/fig_084.png)  

![fig_085](docs/images/fig_085.png)  

![fig_086](docs/images/fig_086.png)  

![fig_087](docs/images/fig_087.png)  

![fig_088](docs/images/fig_088.png)  

![fig_089](docs/images/fig_089.png)  

![fig_090](docs/images/fig_090.png)  

![fig_091](docs/images/fig_091.png)  

![fig_092](docs/images/fig_092.png)  

![fig_093](docs/images/fig_093.png)  

![fig_094](docs/images/fig_094.png)  

![fig_095](docs/images/fig_095.png)  

![fig_096](docs/images/fig_096.png)  

![fig_097](docs/images/fig_097.png)  

![fig_098](docs/images/fig_098.png)  

![fig_099](docs/images/fig_099.png)  

![fig_100](docs/images/fig_100.png)  

![fig_101](docs/images/fig_101.png)  

![fig_102](docs/images/fig_102.png)  

![fig_103](docs/images/fig_103.png)  

![fig_104](docs/images/fig_104.png)  

![fig_105](docs/images/fig_105.png)  

![fig_106](docs/images/fig_106.png)  

![fig_107](docs/images/fig_107.png)  

![fig_108](docs/images/fig_108.png)  

![fig_109](docs/images/fig_109.png)  

![fig_110](docs/images/fig_110.png)  

![fig_111](docs/images/fig_111.png)  

![fig_112](docs/images/fig_112.png)  

![fig_113](docs/images/fig_113.png)  

![fig_114](docs/images/fig_114.png)  

![fig_115](docs/images/fig_115.png)  

![fig_116](docs/images/fig_116.png)  

![fig_117](docs/images/fig_117.png)  

![fig_118](docs/images/fig_118.png)  

![fig_119](docs/images/fig_119.png)  

![fig_120](docs/images/fig_120.png)  

![fig_121](docs/images/fig_121.png)  

![fig_122](docs/images/fig_122.png)  

![fig_123](docs/images/fig_123.png)  

![fig_124](docs/images/fig_124.png)  

![fig_125](docs/images/fig_125.png)  

![fig_126](docs/images/fig_126.png)  

![fig_127](docs/images/fig_127.png)  

![fig_128](docs/images/fig_128.png)  

![fig_129](docs/images/fig_129.png)  

### Distributions by Sub-region

Box plots for sub-regions.

![fig_130](docs/images/fig_130.png)  

![fig_131](docs/images/fig_131.png)  

![fig_132](docs/images/fig_132.png)  

![fig_133](docs/images/fig_133.png)  

![fig_134](docs/images/fig_134.png)  

![fig_135](docs/images/fig_135.png)  

![fig_136](docs/images/fig_136.png)  

![fig_137](docs/images/fig_137.png)  

![fig_138](docs/images/fig_138.png)  

![fig_139](docs/images/fig_139.png)  

![fig_140](docs/images/fig_140.png)  

![fig_141](docs/images/fig_141.png)  

![fig_142](docs/images/fig_142.png)  

![fig_143](docs/images/fig_143.png)  

![fig_144](docs/images/fig_144.png)  

![fig_145](docs/images/fig_145.png)  

![fig_146](docs/images/fig_146.png)  

![fig_147](docs/images/fig_147.png)  

![fig_148](docs/images/fig_148.png)  

![fig_149](docs/images/fig_149.png)  

![fig_150](docs/images/fig_150.png)  

![fig_151](docs/images/fig_151.png)  

![fig_152](docs/images/fig_152.png)  

![fig_153](docs/images/fig_153.png)  

![fig_154](docs/images/fig_154.png)  

![fig_155](docs/images/fig_155.png)  

![fig_156](docs/images/fig_156.png)  

![fig_157](docs/images/fig_157.png)  

![fig_158](docs/images/fig_158.png)  

![fig_159](docs/images/fig_159.png)  

![fig_160](docs/images/fig_160.png)  

![fig_161](docs/images/fig_161.png)  

![fig_162](docs/images/fig_162.png)  

![fig_163](docs/images/fig_163.png)  

![fig_164](docs/images/fig_164.png)  

![fig_165](docs/images/fig_165.png)  

![fig_166](docs/images/fig_166.png)  

![fig_167](docs/images/fig_167.png)  

![fig_168](docs/images/fig_168.png)  

![fig_169](docs/images/fig_169.png)  

![fig_170](docs/images/fig_170.png)  

![fig_171](docs/images/fig_171.png)  

![fig_172](docs/images/fig_172.png)  

![fig_173](docs/images/fig_173.png)  

![fig_174](docs/images/fig_174.png)  

![fig_175](docs/images/fig_175.png)  

![fig_176](docs/images/fig_176.png)  

![fig_177](docs/images/fig_177.png)  

![fig_178](docs/images/fig_178.png)  

![fig_179](docs/images/fig_179.png)  

![fig_180](docs/images/fig_180.png)  

![fig_181](docs/images/fig_181.png)  

![fig_182](docs/images/fig_182.png)  

![fig_183](docs/images/fig_183.png)  

![fig_184](docs/images/fig_184.png)  

![fig_185](docs/images/fig_185.png)  

![fig_186](docs/images/fig_186.png)  

![fig_187](docs/images/fig_187.png)  

![fig_188](docs/images/fig_188.png)  

![fig_189](docs/images/fig_189.png)  

![fig_190](docs/images/fig_190.png)  

![fig_191](docs/images/fig_191.png)  

![fig_192](docs/images/fig_192.png)  

### Other Insights

More plots from the analysis.

![fig_193](docs/images/fig_193.png)  

![fig_194](docs/images/fig_194.png)  

![fig_195](docs/images/fig_195.png)  

![fig_196](docs/images/fig_196.png)  

![fig_197](docs/images/fig_197.png)  

![fig_198](docs/images/fig_198.png)  

![fig_199](docs/images/fig_199.png)  

![fig_200](docs/images/fig_200.png)  

![fig_201](docs/images/fig_201.png)  

![fig_202](docs/images/fig_202.png)  

![fig_203](docs/images/fig_203.png)  

![fig_204](docs/images/fig_204.png)  

![fig_205](docs/images/fig_205.png)  

![fig_206](docs/images/fig_206.png)  

![fig_207](docs/images/fig_207.png)  

![fig_208](docs/images/fig_208.png)  

![fig_209](docs/images/fig_209.png)  

![fig_210](docs/images/fig_210.png)  

![fig_211](docs/images/fig_211.png)  

![fig_212](docs/images/fig_212.png)  

![fig_213](docs/images/fig_213.png)  

![fig_214](docs/images/fig_214.png)  

![fig_215](docs/images/fig_215.png)  

![fig_216](docs/images/fig_216.png)  

![fig_217](docs/images/fig_217.png)  

![fig_218](docs/images/fig_218.png)  

![fig_219](docs/images/fig_219.png)  

![fig_220](docs/images/fig_220.png)  

![fig_221](docs/images/fig_221.png)  

![fig_222](docs/images/fig_222.png)  

![fig_223](docs/images/fig_223.png)  

![fig_224](docs/images/fig_224.png)  

![fig_225](docs/images/fig_225.png)  

## The End: What We Learned

Through this story, we cleaned the data and explored imports from African countries. We saw trends over time, differences by country and product, and distributions. This helps understand trade patterns. If you have questions, the figures are all here!

## Bar Plots of Total Import Value by Country

These bar charts show the total import value for different countries.

![fig_231](docs/images/fig_231.png)  
Bar chart of total import value for a country.  
![fig_232](docs/images/fig_232.png)  
Bar chart for another country.  
![fig_233](docs/images/fig_233.png)  
Bar chart for a different country.  
![fig_234](docs/images/fig_234.png)  
Bar chart showing import value by country.  
![fig_235](docs/images/fig_235.png)  
Another country's import value.  
![fig_236](docs/images/fig_236.png)  
Bar chart for import value.  
![fig_237](docs/images/fig_237.png)  
Country-wise import value.  
![fig_238](docs/images/fig_238.png)  
Bar chart of imports.  
![fig_239](docs/images/fig_239.png)  
Import value by country.  
![fig_240](docs/images/fig_240.png)  
Bar chart for a country.  
![fig_241](docs/images/fig_241.png)  
Bar chart.  
![fig_242](docs/images/fig_242.png)  
Bar chart.  
![fig_243](docs/images/fig_243.png)  
Bar chart.  
![fig_244](docs/images/fig_244.png)  
Bar chart.  
![fig_245](docs/images/fig_245.png)  
Bar chart.  
![fig_246](docs/images/fig_246.png)  
Bar chart of import value.  
![fig_247](docs/images/fig_247.png)  
Bar chart.  
![fig_248](docs/images/fig_248.png)  
Bar chart.  
![fig_249](docs/images/fig_249.png)  
Bar chart.  
![fig_250](docs/images/fig_250.png)  
Bar chart.  
![fig_251](docs/images/fig_251.png)  
Bar chart.  
![fig_252](docs/images/fig_252.png)  
Bar chart.  
![fig_253](docs/images/fig_253.png)  
Bar chart.  
![fig_254](docs/images/fig_254.png)  
Bar chart.  
![fig_255](docs/images/fig_255.png)  
Bar chart.  
![fig_256](docs/images/fig_256.png)  
Bar chart.  
![fig_257](docs/images/fig_257.png)  
Bar chart.  
![fig_258](docs/images/fig_258.png)  
Bar chart.  
![fig_259](docs/images/fig_259.png)  
Bar chart.  
![fig_260](docs/images/fig_260.png)  
Bar chart.  
![fig_261](docs/images/fig_261.png)  
Bar chart.

## Other Plots

### Scatter Plot of Quantity vs Value
![fig_004](docs/images/fig_004.png)  
Scatter plot showing relationship between import quantity and value.

### Bar Plot of Value by Commodity
![fig_005](docs/images/fig_005.png)  
Bar chart of import value by commodity.  
![fig_006](docs/images/fig_006.png)  
Bar chart.  
![fig_007](docs/images/fig_007.png)  
Bar chart.  
![fig_008](docs/images/fig_008.png)  
Bar chart.  
![fig_009](docs/images/fig_009.png)  
Bar chart.  
![fig_010](docs/images/fig_010.png)  
Bar chart.  
![fig_011](docs/images/fig_011.png)  
Bar chart.  
![fig_012](docs/images/fig_012.png)  
Bar chart.  
![fig_013](docs/images/fig_013.png)  
Bar chart.  
![fig_014](docs/images/fig_014.png)  
Bar chart.  
![fig_016](docs/images/fig_016.png)  
Bar chart.  
![fig_017](docs/images/fig_017.png)  
Bar chart.  
![fig_018](docs/images/fig_018.png)  
Bar chart.  
![fig_019](docs/images/fig_019.png)  
Bar chart.  
![fig_020](docs/images/fig_020.png)  
Bar chart.

## Box Plots of Import Values by Country

These box plots show the distribution of import values for each country.

![fig_021](docs/images/fig_021.png)  
Box plot for a country's import values.  
![fig_022](docs/images/fig_022.png)  
Box plot.  
![fig_023](docs/images/fig_023.png)  
Box plot.  
![fig_024](docs/images/fig_024.png)  
Box plot.  
![fig_025](docs/images/fig_025.png)  
Box plot.  
![fig_026](docs/images/fig_026.png)  
Box plot.  
![fig_027](docs/images/fig_027.png)  
Box plot.  
![fig_028](docs/images/fig_028.png)  
Box plot.  
![fig_029](docs/images/fig_029.png)  
Box plot.  
![fig_030](docs/images/fig_030.png)  
Box plot.  
![fig_031](docs/images/fig_031.png)  
Box plot.  
![fig_032](docs/images/fig_032.png)  
Box plot.  
![fig_033](docs/images/fig_033.png)  
Box plot.  
![fig_034](docs/images/fig_034.png)  
Box plot.  
![fig_035](docs/images/fig_035.png)  
Box plot.  
![fig_036](docs/images/fig_036.png)  
Box plot.  
![fig_037](docs/images/fig_037.png)  
Box plot.  
![fig_038](docs/images/fig_038.png)  
Box plot.  
![fig_039](docs/images/fig_039.png)  
Box plot.  
![fig_040](docs/images/fig_040.png)  
Box plot.  
![fig_041](docs/images/fig_041.png)  
Box plot.  
![fig_042](docs/images/fig_042.png)  
Box plot.  
![fig_043](docs/images/fig_043.png)  
Box plot.  
![fig_044](docs/images/fig_044.png)  
Box plot.  
![fig_045](docs/images/fig_045.png)  
Box plot.  
![fig_046](docs/images/fig_046.png)  
Box plot.  
![fig_047](docs/images/fig_047.png)  
Box plot.  
![fig_048](docs/images/fig_048.png)  
Box plot.  
![fig_049](docs/images/fig_049.png)  
Box plot.  
![fig_050](docs/images/fig_050.png)  
Box plot.  
![fig_051](docs/images/fig_051.png)  
Box plot.  
![fig_052](docs/images/fig_052.png)  
Box plot.  
![fig_053](docs/images/fig_053.png)  
Box plot.  
![fig_054](docs/images/fig_054.png)  
Box plot.  
![fig_055](docs/images/fig_055.png)  
Box plot.  
![fig_056](docs/images/fig_056.png)  
Box plot.  
![fig_057](docs/images/fig_057.png)  
Box plot.  
![fig_058](docs/images/fig_058.png)  
Box plot.  
![fig_059](docs/images/fig_059.png)  
Box plot.  
![fig_060](docs/images/fig_060.png)  
Box plot.  
![fig_061](docs/images/fig_061.png)  
Box plot.  
![fig_062](docs/images/fig_062.png)  
Box plot.  
![fig_063](docs/images/fig_063.png)  
Box plot.  
![fig_064](docs/images/fig_064.png)  
Box plot.  
![fig_065](docs/images/fig_065.png)  
Box plot.  
![fig_066](docs/images/fig_066.png)  
Box plot.  
![fig_067](docs/images/fig_067.png)  
Box plot.  
![fig_068](docs/images/fig_068.png)  
Box plot.

## Box Plots of Import Values by Commodity

These box plots show the distribution of import values for each commodity.

![fig_069](docs/images/fig_069.png)  
Box plot for a commodity's import values.  
![fig_070](docs/images/fig_070.png)  
Box plot.  
![fig_071](docs/images/fig_071.png)  
Box plot.  
![fig_072](docs/images/fig_072.png)  
Box plot.  
![fig_073](docs/images/fig_073.png)  
Box plot.  
![fig_074](docs/images/fig_074.png)  
Box plot.  
![fig_075](docs/images/fig_075.png)  
Box plot.  
![fig_076](docs/images/fig_076.png)  
Box plot.  
![fig_077](docs/images/fig_077.png)  
Box plot.  
![fig_078](docs/images/fig_078.png)  
Box plot.  
![fig_079](docs/images/fig_079.png)  
Box plot.  
![fig_080](docs/images/fig_080.png)  
Box plot.  
![fig_081](docs/images/fig_081.png)  
Box plot.  
![fig_082](docs/images/fig_082.png)  
Box plot.  
![fig_083](docs/images/fig_083.png)  
Box plot.  
![fig_084](docs/images/fig_084.png)  
Box plot.  
![fig_085](docs/images/fig_085.png)  
Box plot.  
![fig_086](docs/images/fig_086.png)  
Box plot.  
![fig_087](docs/images/fig_087.png)  
Box plot.  
![fig_088](docs/images/fig_088.png)  
Box plot.  
![fig_089](docs/images/fig_089.png)  
Box plot.  
![fig_090](docs/images/fig_090.png)  
Box plot.  
![fig_091](docs/images/fig_091.png)  
Box plot.  
![fig_092](docs/images/fig_092.png)  
Box plot.  
![fig_093](docs/images/fig_093.png)  
Box plot.  
![fig_094](docs/images/fig_094.png)  
Box plot.  
![fig_095](docs/images/fig_095.png)  
Box plot.  
![fig_096](docs/images/fig_096.png)  
Box plot.  
![fig_097](docs/images/fig_097.png)  
Box plot.  
![fig_098](docs/images/fig_098.png)  
Box plot.  
![fig_099](docs/images/fig_099.png)  
Box plot.  
![fig_100](docs/images/fig_100.png)  
Box plot.  
![fig_101](docs/images/fig_101.png)  
Box plot.  
![fig_102](docs/images/fig_102.png)  
Box plot.  
![fig_103](docs/images/fig_103.png)  
Box plot.  
![fig_104](docs/images/fig_104.png)  
Box plot.  
![fig_105](docs/images/fig_105.png)  
Box plot.  
![fig_106](docs/images/fig_106.png)  
Box plot.  
![fig_107](docs/images/fig_107.png)  
Box plot.  
![fig_108](docs/images/fig_108.png)  
Box plot.  
![fig_109](docs/images/fig_109.png)  
Box plot.  
![fig_110](docs/images/fig_110.png)  
Box plot.  
![fig_111](docs/images/fig_111.png)  
Box plot.  
![fig_112](docs/images/fig_112.png)  
Box plot.  
![fig_113](docs/images/fig_113.png)  
Box plot.  
![fig_114](docs/images/fig_114.png)  
Box plot.  
![fig_115](docs/images/fig_115.png)  
Box plot.  
![fig_116](docs/images/fig_116.png)  
Box plot.  
![fig_117](docs/images/fig_117.png)  
Box plot.  
![fig_118](docs/images/fig_118.png)  
Box plot.  
![fig_119](docs/images/fig_119.png)  
Box plot.  
![fig_120](docs/images/fig_120.png)  
Box plot.  
![fig_121](docs/images/fig_121.png)  
Box plot.  
![fig_122](docs/images/fig_122.png)  
Box plot.  
![fig_123](docs/images/fig_123.png)  
Box plot.  
![fig_124](docs/images/fig_124.png)  
Box plot.  
![fig_125](docs/images/fig_125.png)  
Box plot.  
![fig_126](docs/images/fig_126.png)  
Box plot.  
![fig_127](docs/images/fig_127.png)  
Box plot.  
![fig_128](docs/images/fig_128.png)  
Box plot.  
![fig_129](docs/images/fig_129.png)  
Box plot.

## Box Plots of Import Values by Sub-region

These box plots show the distribution of import values for each sub-region.

![fig_130](docs/images/fig_130.png)  
Box plot for a sub-region's import values.  
![fig_131](docs/images/fig_131.png)  
Box plot.  
![fig_132](docs/images/fig_132.png)  
Box plot.  
![fig_133](docs/images/fig_133.png)  
Box plot.  
![fig_134](docs/images/fig_134.png)  
Box plot.  
![fig_135](docs/images/fig_135.png)  
Box plot.  
![fig_136](docs/images/fig_136.png)  
Box plot.  
![fig_137](docs/images/fig_137.png)  
Box plot.  
![fig_138](docs/images/fig_138.png)  
Box plot.  
![fig_139](docs/images/fig_139.png)  
Box plot.  
![fig_140](docs/images/fig_140.png)  
Box plot.  
![fig_141](docs/images/fig_141.png)  
Box plot.  
![fig_142](docs/images/fig_142.png)  
Box plot.  
![fig_143](docs/images/fig_143.png)  
Box plot.  
![fig_144](docs/images/fig_144.png)  
Box plot.  
![fig_145](docs/images/fig_145.png)  
Box plot.  
![fig_146](docs/images/fig_146.png)  
Box plot.  
![fig_147](docs/images/fig_147.png)  
Box plot.  
![fig_148](docs/images/fig_148.png)  
Box plot.  
![fig_149](docs/images/fig_149.png)  
Box plot.  
![fig_150](docs/images/fig_150.png)  
Box plot.  
![fig_151](docs/images/fig_151.png)  
Box plot.  
![fig_152](docs/images/fig_152.png)  
Box plot.  
![fig_153](docs/images/fig_153.png)  
Box plot.  
![fig_154](docs/images/fig_154.png)  
Box plot.  
![fig_155](docs/images/fig_155.png)  
Box plot.  
![fig_156](docs/images/fig_156.png)  
Box plot.  
![fig_157](docs/images/fig_157.png)  
Box plot.  
![fig_158](docs/images/fig_158.png)  
Box plot.  
![fig_159](docs/images/fig_159.png)  
Box plot.  
![fig_160](docs/images/fig_160.png)  
Box plot.  
![fig_161](docs/images/fig_161.png)  
Box plot.  
![fig_162](docs/images/fig_162.png)  
Box plot.  
![fig_163](docs/images/fig_163.png)  
Box plot.  
![fig_164](docs/images/fig_164.png)  
Box plot.  
![fig_165](docs/images/fig_165.png)  
Box plot.  
![fig_166](docs/images/fig_166.png)  
Box plot.  
![fig_167](docs/images/fig_167.png)  
Box plot.  
![fig_168](docs/images/fig_168.png)  
Box plot.  
![fig_169](docs/images/fig_169.png)  
Box plot.  
![fig_170](docs/images/fig_170.png)  
Box plot.  
![fig_171](docs/images/fig_171.png)  
Box plot.  
![fig_172](docs/images/fig_172.png)  
Box plot.  
![fig_173](docs/images/fig_173.png)  
Box plot.  
![fig_174](docs/images/fig_174.png)  
Box plot.  
![fig_175](docs/images/fig_175.png)  
Box plot.  
![fig_176](docs/images/fig_176.png)  
Box plot.  
![fig_177](docs/images/fig_177.png)  
Box plot.  
![fig_178](docs/images/fig_178.png)  
Box plot.  
![fig_179](docs/images/fig_179.png)  
Box plot.  
![fig_180](docs/images/fig_180.png)  
Box plot.  
![fig_181](docs/images/fig_181.png)  
Box plot.  
![fig_182](docs/images/fig_182.png)  
Box plot.  
![fig_183](docs/images/fig_183.png)  
Box plot.  
![fig_184](docs/images/fig_184.png)  
Box plot.  
![fig_185](docs/images/fig_185.png)  
Box plot.  
![fig_186](docs/images/fig_186.png)  
Box plot.  
![fig_187](docs/images/fig_187.png)  
Box plot.  
![fig_188](docs/images/fig_188.png)  
Box plot.  
![fig_189](docs/images/fig_189.png)  
Box plot.  
![fig_190](docs/images/fig_190.png)  
Box plot.  
![fig_191](docs/images/fig_191.png)  
Box plot.  
![fig_192](docs/images/fig_192.png)  
Box plot.

## Additional Figures

These are other plots from the analysis.

![fig_193](docs/images/fig_193.png)  
Plot.  
![fig_194](docs/images/fig_194.png)  
Plot.  
![fig_195](docs/images/fig_195.png)  
Plot.  
![fig_196](docs/images/fig_196.png)  
Plot.  
![fig_197](docs/images/fig_197.png)  
Plot.  
![fig_198](docs/images/fig_198.png)  
Plot.  
![fig_199](docs/images/fig_199.png)  
Plot.  
![fig_200](docs/images/fig_200.png)  
Plot.  
![fig_201](docs/images/fig_201.png)  
Plot.  
![fig_202](docs/images/fig_202.png)  
Plot.  
![fig_203](docs/images/fig_203.png)  
Plot.  
![fig_204](docs/images/fig_204.png)  
Plot.  
![fig_205](docs/images/fig_205.png)  
Plot.  
![fig_206](docs/images/fig_206.png)  
Plot.  
![fig_207](docs/images/fig_207.png)  
Plot.  
![fig_208](docs/images/fig_208.png)  
Plot.  
![fig_209](docs/images/fig_209.png)  
Plot.  
![fig_210](docs/images/fig_210.png)  
Plot.  
![fig_211](docs/images/fig_211.png)  
Plot.  
![fig_212](docs/images/fig_212.png)  
Plot.  
![fig_213](docs/images/fig_213.png)  
Plot.  
![fig_214](docs/images/fig_214.png)  
Plot.  
![fig_215](docs/images/fig_215.png)  
Plot.  
![fig_216](docs/images/fig_216.png)  
Plot.  
![fig_217](docs/images/fig_217.png)  
Plot.  
![fig_218](docs/images/fig_218.png)  
Plot.  
![fig_219](docs/images/fig_219.png)  
Plot.  
![fig_220](docs/images/fig_220.png)  
Plot.  
![fig_221](docs/images/fig_221.png)  
Plot.  
![fig_222](docs/images/fig_222.png)  
Plot.  
![fig_223](docs/images/fig_223.png)  
Plot.  
![fig_224](docs/images/fig_224.png)  
Plot.  
![fig_225](docs/images/fig_225.png)  
Plot.

Why it matters: a small set of commodities often drives most value; these are high-impact categories.

What to do next: drill into each top commodity to see seasonality and major trading partners.

---

## 5. Commodity composition for a country (example) — fig_134.png
![fig_134](docs/images/fig_134.png)

What you see: pie chart showing how a single country's imports are split across the top commodities.

Why it matters: tells you what a country's import economy looks like at a glance.

What to do next: compare this with other countries to see specialization patterns.

---

## 6. Quantity vs USD value scatter — fig_240.png
![fig_240](docs/images/fig_240.png)

What you see: scatter of shipment quantity (x) versus USD value (y). May be log-scaled.

Why it matters: you can see relationships between quantity and price — e.g., expensive items (high USD, low quantity) vs cheap bulk items (high quantity, low USD).

What to do next: compute price-per-unit and flag extreme prices for investigation.

---

## 7. Model diagnostics (Actual vs Predicted) — fig_193.png
![fig_193](docs/images/fig_193.png)

What you see: predicted values plotted against actual values; a line shows perfect prediction.

Why it matters: closeness to the line means the model predicts well; deviations show where the model fails.

What to do next: if residuals are large for big USD values, try log-transforming the target or use tree-based models.

---

## 8. Anomaly detection summary — fig_225.png
![fig_225](docs/images/fig_225.png)

What you see: a plot showing anomaly scores and highlighted top anomalies; the notebook also prints the top anomalous rows.

Why it matters: quickly find suspicious shipments (possible errors or unusual trades).

What to do next: review the top anomalous rows and decide whether to remove, correct, or treat them separately.

---

If this is the kind of simple, visual-first doc you want, I can:
- Embed more figures (all 261) in this simple format (will be long).
- Replace each figure with a country/commodity-labeled filename (semantic names) so the images are self-explanatory.
- Expand any figure's explanation into a short paragraph with exact code snippets from the notebook.

Tell me which of these three you'd like next (embed all / semantic rename / expand examples), or list specific figures to expand.
