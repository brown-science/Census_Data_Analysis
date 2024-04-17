#### This R markdown file contains an analysis of US census data from 1996. I recommend implementing a logistic regression model to predict whether individuals, based on the census variables provided, make over $50,000/year. I also explored an initial random forest model, which has the potential to significantly outperform the logistic regression model once tuned, but to maintain interpretability, I prioritized the logistic regression model.

Before preparing the data for analysis, the following packages must be
loaded.

    library(DBI)
    library(corrplot)
    library(tidyverse)
    library(DataExplorer)
    library(caret)
    library(glue)
    library(car)
    library(rpart)
    library(rpart.plot)
    library(Hmisc)
    library(ROCit)
    library(randomForest)
    library(DBI)

To flatten the SQLite database into a data frame for analysis, I’ll read
the database into R using RSQLite.

    mydb <- dbConnect(RSQLite::SQLite(), "census_data.sqlite")
    dbGetQuery(mydb, "SELECT COUNT(*) FROM records;") 

    ##   COUNT(*)
    ## 1    48842

The database is now loaded as “mydb,” with 48,842 observations. Now,
I’ll flatten the database into a data frame using a series of joins and
close the SQLite connection.

    # If the view already exists, run the line below
    #dbExecute(mydb, "DROP VIEW [records_with_names]")
    # This query creates a view where the country_id, education_level_id, etc. are replaced 
    # in place by the country name, education_level name, etc., through a series of joins 
    dbExecute(mydb, " CREATE VIEW records_with_names AS 
                      SELECT
                          records.id,
                          records.age,
                          records.education_num,
                          records.capital_gain,
                          records.capital_loss,
                          records.hours_week,
                          records.over_50k, 
                          countries.name country_name,
                          education_levels.name education_level,
                          marital_statuses.name marital_status,
                          occupations.name occupation,
                          races.name race,
                          relationships.name relationship,
                          sexes.name sex,
                          workclasses.name workclass
                      FROM records
                      JOIN countries ON records.country_id = countries.id
                      JOIN education_levels ON records.education_level_id = education_levels.id
                      JOIN marital_statuses ON records.marital_status_id = marital_statuses.id
                      JOIN occupations ON records.occupation_id = occupations.id
                      JOIN races ON records.race_id = races.id
                      JOIN relationships ON records.relationship_id = relationships.id
                      JOIN sexes ON records.sex_id = sexes.id
                      JOIN workclasses ON records.workclass_id = workclasses.id; ") 

    data <- as.data.frame(dbGetQuery(mydb, "SELECT * FROM records_with_names ;")) # saves the view as "data"
    write.csv(data, file = "data.csv", row.names = FALSE)
    dbExecute(mydb, "DROP VIEW [records_with_names]") # drops the view to prevent errors when re-running the code from top to bottom
    dbDisconnect(mydb) 

Now that the data is flattened and loaded into the environment, I’ll do
some pre-processing and then Exploratory Data Analysis, beginning with
continuous variables.

    # The country, occupation, and workclass columns had a key corresponding to 
    # "?". I'll replace this character with "Missing"
    data <- data %>%
      mutate_all(~ ifelse(. == "?", "Missing", .))

    # Create vectors for column names by type
    vars <- colnames(data)
    cont_vars <- c("age", "education_num", "capital_gain", "capital_loss", "hours_week")
    cat_vars <- data %>%
      select(-matches(cont_vars), -c(id, over_50k)) %>% # keeps everything but continuous variables, id, and target
      names()

    # Convert categorical variables to factors
    for (var in cat_vars){
      data[[var]] <- as.factor(data[[var]])
    }

    # proportion or 0's and 1's for the target variable
    table(data$over_50k) # 37,155 0's
    37155/nrow(data) # 76% under 50k, 24% over 50k, not a rare event

    # Automated EDA report for continuous variables
    data %>%
      select(cont_vars, over_50k) %>%
          create_report(
            output_file  = "continuous_var_EDA.html",
            y            = "over_50k",
            report_title = "EDA Report - Over 50K census data"
        )

The EDA report (“continuous\_var\_EDA.html”) revealed that most
variables have a right-skewed distribution and education\_num and
hours\_week have clear modes between 8-12 and 40 respectively. Most
importantly, the box plots reveal that age, education\_num, and
hours\_week appear to have a relationship with the response, See the box
blot of hours\_week v. over\_50k below. Generally, individuals who work
more, are more likely to make over 50k annually.

    ggplot(data, aes(x = factor(over_50k), y = hours_week)) +
          geom_boxplot() +
          labs(title = "Box Plot: hours worked per week v. over 50k/year",
               x = "Over 50K",
               y = "hours worked per week")

![](main_files/figure-markdown_strict/unnamed-chunk-5-1.png)

Now let’s look at the categorical variables

    target_var <- "over_50k"
    for (var in cat_vars) {  # Create a bar plot for each categorical variable
      plot_data <- data.frame(x = data[[var]], target = data[[target_var]]) # subset data frame to the variable of interest
      
       p <- ggplot(plot_data, aes(x = x, fill = factor(target))) +
        geom_bar(stat = "count", position = "stack", color = "black") + # stacked bar plots with black outlines
        labs(title = paste("Distribution of", var, "by", target_var),
             x = var,
             y = "Observation Count",
             fill = target_var) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))   # Adjust the angle 
         # change background color
      # Save the plot to the EDA folder
      ggsave(paste("EDA/cat_v_target_", gsub(" ", "_", var), ".png", sep = ""), plot = p)
      
      if(var == "marital_status"){print(p)} # display the marital_status plot as an example in the html file
    }

![](main_files/figure-markdown_strict/unnamed-chunk-6-1.png)

These plots show the distribution of the categorical variables and their
relationship with the target. Some interesting relationships are that
married individuals have a higher chance of making over 50k a year than
all other categories with a reasonable sample size.

Now I’ll move on to logistic regression model considerations

    data <- data %>% select(-id) #drop  id
    # First I'll split the data into train/validation/ and test: 70/20/10 and check for any separation concerns
    set.seed(12345)
    trainIndex <- createDataPartition(data$over_50k, p = 0.7, list = FALSE)
    # Split into train and validation_test
    train <- data[trainIndex, ]
    validation_test <- data[-trainIndex, ]
    # Further split validation_test into validation and test
    validationIndex <- createDataPartition(validation_test$over_50k, p = 0.67, list = FALSE)
    validation <- validation_test[-validationIndex, ]
    test <- validation_test[validationIndex, ]

    sep_conc = c() # Will be populated if there are any separation concerns
    count = 1 # index for above vector, used in loop

    for (i in 1:length(train)){ # The first few lines of the loop prints the tables with variable comparisons.
      
      print(glue("Table of {colnames(train)[i]} (left) v. over_50k (right)")) # prints the variable comparisons above each table
      print(table(train[, i], train$over_50k)) # prints the actual table
      cat("\n") # adds a space between each table
      if(0 %in% table(train[, i], train$over_50k)){ # If there are separation concerns (an observation value perfectly predicts the target)
        sep_conc[count] = colnames(train)[i] # append the variable to this list at position "count"
        count = count + 1 # count increases by one after each variable is added to sep_conc
      }
    }

    ## Table of age (left) v. over_50k (right)
    ##     
    ##        0   1
    ##   17 410   0
    ##   18 601   0
    ##   19 709   3
    ##   20 776   1
    ##   21 778   3
    ##   22 799  12
    ##   23 925  12
    ##   24 814  36
    ##   25 764  53
    ##   26 747  57
    ##   27 785  90
    ##   28 794 120
    ##   29 728 139
    ##   30 711 180
    ##   31 750 193
    ##   32 637 207
    ##   33 721 208
    ##   34 675 247
    ##   35 708 238
    ##   36 674 290
    ##   37 579 302
    ##   38 606 294
    ##   39 537 293
    ##   40 546 264
    ##   41 569 300
    ##   42 519 297
    ##   43 476 281
    ##   44 471 288
    ##   45 484 307
    ##   46 458 302
    ##   47 462 295
    ##   48 361 234
    ##   49 368 238
    ##   50 341 253
    ##   51 373 241
    ##   52 323 208
    ##   53 287 199
    ##   54 285 178
    ##   55 280 148
    ##   56 272 137
    ##   57 246 148
    ##   58 261 125
    ##   59 221 138
    ##   60 218  98
    ##   61 209  93
    ##   62 191  71
    ##   63 157  57
    ##   64 168  65
    ##   65 160  50
    ##   66 128  44
    ##   67 131  41
    ##   68  87  28
    ##   69  84  22
    ##   70  73  23
    ##   71  63  20
    ##   72  68  11
    ##   73  68  14
    ##   74  39  12
    ##   75  42   9
    ##   76  42   7
    ##   77  32   6
    ##   78  23   3
    ##   79  12   4
    ##   80  22   1
    ##   81  22   7
    ##   82  13   0
    ##   83   5   1
    ##   84   6   1
    ##   85   4   0
    ##   86   1   0
    ##   87   2   0
    ##   88   3   1
    ##   90  30   8
    ## 
    ## Table of education_num (left) v. over_50k (right)
    ##     
    ##         0    1
    ##   1    61    0
    ##   2   155    5
    ##   3   318   22
    ##   4   623   38
    ##   5   491   30
    ##   6   916   52
    ##   7  1185   58
    ##   8   406   34
    ##   9  9308 1752
    ##   10 6144 1444
    ##   11 1060  382
    ##   12  816  296
    ##   13 3340 2362
    ##   14  831 1020
    ##   15  166  452
    ##   16  114  309
    ## 
    ## Table of capital_gain (left) v. over_50k (right)
    ##        
    ##             0     1
    ##   0     24889  6441
    ##   114       7     0
    ##   401       3     0
    ##   594      33     0
    ##   914       4     0
    ##   991       4     0
    ##   1055     23     0
    ##   1086      5     0
    ##   1111      1     0
    ##   1151      8     0
    ##   1173      4     0
    ##   1264      1     0
    ##   1409      5     0
    ##   1424      1     0
    ##   1455      3     0
    ##   1471      8     0
    ##   1506     16     0
    ##   1639      1     0
    ##   1731      1     0
    ##   1797      9     0
    ##   1831      6     0
    ##   1848      8     0
    ##   2009      1     0
    ##   2036      4     0
    ##   2050      5     0
    ##   2062      2     0
    ##   2105      6     0
    ##   2174     51     0
    ##   2176     20     0
    ##   2202     21     0
    ##   2228      3     0
    ##   2290      8     0
    ##   2329      3     0
    ##   2346      3     0
    ##   2354     15     0
    ##   2387      1     0
    ##   2407     14     0
    ##   2414      7     0
    ##   2463     10     0
    ##   2538      5     0
    ##   2580     15     0
    ##   2597     18     0
    ##   2635     11     0
    ##   2653      8     0
    ##   2829     25     0
    ##   2885     17     0
    ##   2907     15     0
    ##   2936      3     0
    ##   2961      2     0
    ##   2964     11     0
    ##   2977      7     0
    ##   2993      3     0
    ##   3103      7   103
    ##   3137     38     0
    ##   3273      4     0
    ##   3325     54     0
    ##   3411     28     0
    ##   3418      6     0
    ##   3432      2     0
    ##   3456      3     0
    ##   3464     17     0
    ##   3471     10     0
    ##   3674     15     0
    ##   3781     12     0
    ##   3818      7     0
    ##   3887      7     0
    ##   3908     27     0
    ##   3942     11     0
    ##   4064     35     0
    ##   4101     21     0
    ##   4386     10    65
    ##   4416     17     0
    ##   4508     17     0
    ##   4650     42     0
    ##   4687      0     3
    ##   4787      0    24
    ##   4865     15     0
    ##   4931      3     0
    ##   4934      0    10
    ##   5013     82     0
    ##   5060      2     0
    ##   5178      0   110
    ##   5455     11     0
    ##   5556      0     6
    ##   5721      6     0
    ##   6097      0     2
    ##   6360      1     0
    ##   6418      0    11
    ##   6497     11     0
    ##   6514      0     8
    ##   6612      0     1
    ##   6723      4     0
    ##   6767      4     0
    ##   6849     32     0
    ##   7262      0     1
    ##   7298      0   252
    ##   7430      0    10
    ##   7443      3     0
    ##   7688      0   302
    ##   7896      0     3
    ##   7978      2     0
    ##   8614      0    70
    ##   9386      0    26
    ##   9562      0     4
    ##   10520     0    48
    ##   10566     7     0
    ##   10605     0    13
    ##   11678     0     4
    ##   13550     0    30
    ##   14084     0    36
    ##   14344     0    20
    ##   15020     0     7
    ##   15024     0   384
    ##   15831     0     5
    ##   18481     0     1
    ##   20051     0    35
    ##   22040     1     0
    ##   25124     0     5
    ##   25236     0    12
    ##   27828     0    33
    ##   34095     3     0
    ##   41310     3     0
    ##   99999     0   171
    ## 
    ## Table of capital_loss (left) v. over_50k (right)
    ##       
    ##            0     1
    ##   0    25128  7439
    ##   155      1     0
    ##   213      3     0
    ##   323      4     0
    ##   419      3     0
    ##   625     10     0
    ##   653      1     2
    ##   810      2     0
    ##   880      3     0
    ##   974      2     0
    ##   1092     6     0
    ##   1138     2     0
    ##   1258     4     0
    ##   1340     9     0
    ##   1380     5     0
    ##   1408    23     0
    ##   1411     4     0
    ##   1421     1     0
    ##   1429     1     0
    ##   1485    18    26
    ##   1504    21     0
    ##   1510     2     0
    ##   1539     1     0
    ##   1564     0    31
    ##   1573     8     0
    ##   1579    18     0
    ##   1590    43     0
    ##   1594     7     0
    ##   1602    44     0
    ##   1617     5     0
    ##   1628    16     0
    ##   1648     1     1
    ##   1651     7     0
    ##   1668     5     0
    ##   1669    25     0
    ##   1672    39     0
    ##   1719    26     0
    ##   1721    19     0
    ##   1726     7     0
    ##   1735     2     0
    ##   1740    41     0
    ##   1741    30     0
    ##   1755     0     2
    ##   1762    13     0
    ##   1816     2     0
    ##   1825     0     4
    ##   1844     2     0
    ##   1848     0    51
    ##   1870     1     0
    ##   1876    44     0
    ##   1887     0   165
    ##   1902    15   196
    ##   1911     0     1
    ##   1944     2     0
    ##   1974    24     0
    ##   1977     0   181
    ##   1980    24     0
    ##   2001    27     0
    ##   2002    25     0
    ##   2042     9     0
    ##   2051    22     0
    ##   2057    13     0
    ##   2080     1     0
    ##   2129     4     0
    ##   2149     4     0
    ##   2163     2     0
    ##   2174     0    10
    ##   2179    14     0
    ##   2201     0     1
    ##   2205    16     0
    ##   2206     5     0
    ##   2231     0     5
    ##   2238     3     0
    ##   2246     0     7
    ##   2258    14    11
    ##   2267     2     0
    ##   2282     0     1
    ##   2339    22     0
    ##   2352     1     0
    ##   2377     7     9
    ##   2392     0     7
    ##   2415     0    51
    ##   2444     0    16
    ##   2457     3     0
    ##   2465     1     0
    ##   2467     1     0
    ##   2472     0     4
    ##   2489     1     0
    ##   2547     0     5
    ##   2559     0    12
    ##   2603     3     0
    ##   2754     2     0
    ##   2824     0    13
    ##   3004     0     4
    ##   3175     1     0
    ##   3683     0     1
    ##   3770     3     0
    ##   3900     2     0
    ##   4356     2     0
    ## 
    ## Table of hours_week (left) v. over_50k (right)
    ##     
    ##          0     1
    ##   1     13     1
    ##   2     27     8
    ##   3     42     1
    ##   4     46     5
    ##   5     60     8
    ##   6     50     9
    ##   7     26     7
    ##   8    141    17
    ##   9     16     3
    ##   10   273    24
    ##   11    16     0
    ##   12   163    10
    ##   13    18     3
    ##   14    36     2
    ##   15   433    23
    ##   16   184    16
    ##   17    27     3
    ##   18    79     4
    ##   19     9     0
    ##   20  1217    74
    ##   21    27     3
    ##   22    38     2
    ##   23    23     1
    ##   24   230    24
    ##   25   630    34
    ##   26    22     4
    ##   27    27     2
    ##   28    85     7
    ##   29     8     1
    ##   30  1112    76
    ##   31     8     2
    ##   32   277    33
    ##   33    39     4
    ##   34    33     4
    ##   35  1160   205
    ##   36   199    42
    ##   37   140    24
    ##   38   404    73
    ##   39    39     7
    ##   40 12514  3449
    ##   41    32    11
    ##   42   173    74
    ##   43   120    41
    ##   44   146    78
    ##   45  1206   703
    ##   46    50    31
    ##   47    38    20
    ##   48   362   177
    ##   49    18     5
    ##   50  1620  1344
    ##   51    12     4
    ##   52   105    44
    ##   53    22     3
    ##   54    23    17
    ##   55   404   346
    ##   56    65    45
    ##   57    11     5
    ##   58    18     8
    ##   59     4     1
    ##   60   865   669
    ##   61     1     2
    ##   62    12     1
    ##   63     8     3
    ##   64    11     5
    ##   65   148   101
    ##   66    11     3
    ##   67     1     3
    ##   68     4     3
    ##   69     0     1
    ##   70   197   113
    ##   72    49    30
    ##   73     1     0
    ##   74     1     0
    ##   75    45    25
    ##   76     1     2
    ##   77     5     0
    ##   78     5     3
    ##   79     1     0
    ##   80    95    48
    ##   81     2     0
    ##   82     1     0
    ##   84    31    17
    ##   85     7     5
    ##   86     1     1
    ##   88     2     0
    ##   89     1     1
    ##   90    18    14
    ##   91     1     0
    ##   92     2     1
    ##   94     1     0
    ##   96     5     1
    ##   97     0     1
    ##   98     7     2
    ##   99    74    29
    ## 
    ## Table of over_50k (left) v. over_50k (right)
    ##    
    ##         0     1
    ##   0 25934     0
    ##   1     0  8256
    ## 
    ## Table of country_name (left) v. over_50k (right)
    ##                             
    ##                                  0     1
    ##   Cambodia                      13     9
    ##   Canada                        84    48
    ##   China                         63    29
    ##   Columbia                      56     3
    ##   Cuba                          68    19
    ##   Dominican-Republic            63     5
    ##   Ecuador                       27     4
    ##   El-Salvador                   93     9
    ##   England                       61    30
    ##   France                        15    10
    ##   Germany                       93    39
    ##   Greece                        24    14
    ##   Guatemala                     64     3
    ##   Haiti                         44     6
    ##   Holand-Netherlands             1     0
    ##   Honduras                      11     2
    ##   Hong                          13     7
    ##   Hungary                        9     5
    ##   India                         62    48
    ##   Iran                          28    16
    ##   Ireland                       21    10
    ##   Italy                         52    25
    ##   Jamaica                       60     9
    ##   Japan                         48    21
    ##   Laos                          11     2
    ##   Mexico                       622    31
    ##   Missing                      424   165
    ##   Nicaragua                     30     2
    ##   Outlying-US(Guam-USVI-etc)    15     1
    ##   Peru                          30     3
    ##   Philippines                  147    58
    ##   Poland                        54    10
    ##   Portugal                      37     7
    ##   Puerto-Rico                  121    15
    ##   Scotland                      14     3
    ##   South                         69    17
    ##   Taiwan                        31    17
    ##   Thailand                      18     2
    ##   Trinadad&Tobago               21     1
    ##   United-States              23155  7538
    ##   Vietnam                       50     6
    ##   Yugoslavia                    12     7
    ## 
    ## Table of education_level (left) v. over_50k (right)
    ##               
    ##                   0    1
    ##   10th          916   52
    ##   11th         1185   58
    ##   12th          406   34
    ##   1st-4th       155    5
    ##   5th-6th       318   22
    ##   7th-8th       623   38
    ##   9th           491   30
    ##   Assoc-acdm    816  296
    ##   Assoc-voc    1060  382
    ##   Bachelors    3340 2362
    ##   Doctorate     114  309
    ##   HS-grad      9308 1752
    ##   Masters       831 1020
    ##   Preschool      61    0
    ##   Prof-school   166  452
    ##   Some-college 6144 1444
    ## 
    ## Table of marital_status (left) v. over_50k (right)
    ##                        
    ##                             0     1
    ##   Divorced               4141   472
    ##   Married-AF-spouse        19    13
    ##   Married-civ-spouse     8683  7047
    ##   Married-spouse-absent   389    46
    ##   Never-married         10765   518
    ##   Separated               996    73
    ##   Widowed                 941    87
    ## 
    ## Table of occupation (left) v. over_50k (right)
    ##                    
    ##                        0    1
    ##   Adm-clerical      3414  526
    ##   Armed-Forces         7    2
    ##   Craft-repair      3275  983
    ##   Exec-managerial   2205 2046
    ##   Farming-fishing    908  118
    ##   Handlers-cleaners 1360   99
    ##   Machine-op-inspct 1829  257
    ##   Missing           1778  180
    ##   Other-service     3303  155
    ##   Priv-house-serv    149    3
    ##   Prof-specialty    2403 1978
    ##   Protective-serv    451  218
    ##   Sales             2821 1054
    ##   Tech-support       716  311
    ##   Transport-moving  1315  326
    ## 
    ## Table of race (left) v. over_50k (right)
    ##                     
    ##                          0     1
    ##   Amer-Indian-Eskimo   264    33
    ##   Asian-Pac-Islander   777   310
    ##   Black               2898   408
    ##   Other                247    38
    ##   White              21748  7467
    ## 
    ## Table of relationship (left) v. over_50k (right)
    ##                 
    ##                     0    1
    ##   Husband        7602 6233
    ##   Not-in-family  7897  906
    ##   Other-relative 1019   34
    ##   Own-child      5204   75
    ##   Unmarried      3338  222
    ##   Wife            874  786
    ## 
    ## Table of sex (left) v. over_50k (right)
    ##         
    ##              0     1
    ##   Female 10068  1268
    ##   Male   15866  6988
    ## 
    ## Table of workclass (left) v. over_50k (right)
    ##                   
    ##                        0     1
    ##   Federal-gov        596   395
    ##   Local-gov         1581   641
    ##   Missing           1773   180
    ##   Never-worked         5     0
    ##   Private          18474  5240
    ##   Self-emp-inc       544   682
    ##   Self-emp-not-inc  1948   755
    ##   State-gov         1002   362
    ##   Without-pay         11     1

    print(sep_conc) # The separation concerns for country_name, education_num, workclass, and education_level can easily be corrected by collapsing categories or removing an outlier(country_name).

    ## [1] "age"             "education_num"   "capital_gain"    "capital_loss"   
    ## [5] "hours_week"      "over_50k"        "country_name"    "education_level"
    ## [9] "workclass"

    train <- train[train$country_name != "Holand-Netherlands", ] # remove single observation from Holand-Netherlands

    train$education_num[which(train$education_num < 3)] <- "1-2" # collapses 1 and 2 into a single category
    train$education_num <- factor(train$education_num) # This variable is now categorical

    train$workclass[which(train$workclass == "Never-worked")] <- "Without-pay" # Converts Never-Worked observations to Without-pay 
    train$workclass[which(train$workclass == "Without-pay")] <- "Never-worked-or-Without-pay" # These categories are now one
    train$workclass <- factor(train$workclass) # re-factor

    train$education_level[which(train$education_num == "Preschool")] <- "4th-and-below" # Converts Preschool observations to 4th-and-below
    train$education_level[which(train$education_num == "1st-4th")] <- "4th-and-below"  # Converts 1st-4th observations to 4th-and-below
    train$education_level <- factor(train$education_level) # re-factor

Handling separation concerns with continuous variables - The continuous
variables with separation concerns are age, capital\_gain,
capital\_loss, and hours\_week. I’ll create bins for these variables
using decision trees

    age_tree <- rpart(over_50k ~ age, data = train, method = "anova")
    prp(age_tree)

![](main_files/figure-markdown_strict/unnamed-chunk-8-1.png)

    #Creating bins for age variable
    breaks <- c(0, 29, Inf)
    train$age <- cut(train$age, breaks = breaks, labels = c("<30", "30+"), include.lowest = TRUE, right = FALSE)

    capital_gain_tree <- rpart(over_50k ~ capital_gain, data = train, method = "anova")
    prp(capital_gain_tree)

![](main_files/figure-markdown_strict/unnamed-chunk-8-2.png)

    #Creating bins for capital_gain variable
    breaks <- c(0, 5118, Inf)
    train$capital_gain <- cut(train$capital_gain, breaks = breaks, labels = c("<5119", "5119+"), include.lowest = TRUE, right = FALSE)


    capital_loss_tree <- rpart(over_50k ~ capital_loss, data = train, method = "anova")
    prp(capital_loss_tree)

![](main_files/figure-markdown_strict/unnamed-chunk-8-3.png)

    #Creating bins for capital_loss variable
    breaks <- c(0, 1820, Inf)
    train$capital_loss <- cut(train$capital_loss, breaks = breaks, labels = c("<1821", "1821+"), include.lowest = TRUE, right = FALSE)

    hours_week_tree <- rpart(over_50k ~ hours_week, data = train, method = "anova")
    prp(hours_week_tree)

![](main_files/figure-markdown_strict/unnamed-chunk-8-4.png)

    #Creating bins for hours_week variable
    breaks <- c(0, 34, 43, Inf)
    train$hours_week <- cut(train$hours_week, breaks = breaks, labels = c("<35","35-43", "44+"), include.lowest = TRUE, right = FALSE)

Since all continuous variables are now binned, I can avoid the linearity
of the log(odds) assumption and can move on to modeling. I chose to
select variables through backward selection using AIC.

    full.model <- glm(factor(over_50k) ~ ., data = train, family = binomial(link = "logit")) # Full model contains all variables

    empty.model <- glm(factor(over_50k) ~ 1, data = train, family = binomial(link = "logit")) # Intercept only


    step_model <- step(full.model, trace = F, # backwards stepwise selection using AIC
                       scope = list(lower = empty.model, upper = full.model),
                       direction = "backward") 

    AIC(step_model)# 20,959.67

    ## [1] 20959.67

    # Sorting variables by p-value
    pval_df <- rownames_to_column(as.data.frame(Anova(step_model)[3]))
    colnames(pval_df) = c("variable", "pval")
    pval_df <- pval_df %>% arrange(pval)

    # gvifs <- vif(step_model)[,1] # checks model for collinearity concerns

    ###########################
    # When step_model is passed to gvifs, I get an error that there are aliases coefficients.  
    # I dropped marital_status and workclass and got a worse AIC, but I can trust the coefficients now
    ###########################

    new_model <- glm(factor(over_50k) ~ age + education_num + capital_gain + capital_loss + 
                     hours_week + country_name + occupation + race + relationship + sex,
                     data = train, family = binomial(link = "logit"))

    AIC(new_model)# 21,174.03

    ## [1] 21174.03

    gvifs <- vif(new_model)[,1]
    gvifs <- sort(gvifs, decreasing = T)
    print(gvifs) # collinearity issue solved, all gvifs < 5

    ##  relationship  country_name           sex          race    occupation 
    ##      3.298334      3.198232      2.861104      2.718267      2.098448 
    ## education_num    hours_week           age  capital_gain  capital_loss 
    ##      1.942396      1.183772      1.048156      1.033788      1.010181

    pval_df <- rownames_to_column(as.data.frame(Anova(new_model)[3])) # extracts p-value
    colnames(pval_df) = c("variable", "pval")
    pval_df <- pval_df %>% arrange(pval)

    main.effects.model <- new_model 

The power of logistic regression is the interpretable coefficients.
Exponentiating the coefficients and performing the below transformations
gives the percent change in odds from the reference level to the next
level of a variable.

    #summary(main.effects.model) # gives the model summary

    # the coefficient estimate for age30+ = 1.30728
    (exp(1.30728)-1) * 100 

    ## [1] 269.6107

Someone who is 30 years old or older has 269.61% higher odds of making
over 50K a year compared to someone who is 29 years old or younger.

Now that I’ve selected a final model, it’s time to test it on the
hold-out data. Since I didn’t validate more than one model, I’ll roll up
the validation and test sets. I’ll need to bin the variables in the same
way I binned the training variables

    test_set <- rbind(validation, test) # combine validation and test

    # Making the same transformations as in the train set 
    test_set$education_num[which(test_set$education_num < 3)] <- "1-2"
    test_set$education_num <- factor(test_set$education_num)

    test_set$workclass[which(test_set$workclass == "Never-worked")] <- "Without-pay"
    test_set$workclass[which(test_set$workclass == "Without-pay")] <- "Never-worked-or-Without-pay"
    test_set$workclass <- factor(test_set$workclass)

    test_set$education_level[which(test_set$education_num == "Preschool")] <- "4th-and-below"
    test_set$education_level[which(test_set$education_num == "1at-4th")] <- "4th-and-below"
    test_set$education_level <- factor(test_set$education_level)

    #Creating bins for age variable
    breaks <- c(0, 29, Inf)
    test_set$age <- cut(test_set$age, breaks = breaks, labels = c("<30", "30+"), include.lowest = TRUE, right = FALSE)
    #Creating bins for capital_gain variable
    breaks <- c(0, 5118, Inf)
    test_set$capital_gain <- cut(test_set$capital_gain, breaks = breaks, labels = c("<5119", "5119+"), include.lowest = TRUE, right = FALSE)
    #Creating bins for capital_loss variable
    breaks <- c(0, 1820, Inf)
    test_set$capital_loss <- cut(test_set$capital_loss, breaks = breaks, labels = c("<1821", "1821+"), include.lowest = TRUE, right = FALSE)
    #Creating bins for hours_week variable
    breaks <- c(0, 34, 43, Inf)
    test_set$hours_week <- cut(test_set$hours_week, breaks = breaks, labels = c("<35","35-43", "44+"), include.lowest = TRUE, right = FALSE)

    # Testing the model
    train$p_hat <- predict(main.effects.model, type = "response")
    somers2(train$p_hat, train$over_50k) #91.24% Concordance

    ##            C          Dxy            n      Missing 
    ## 9.124105e-01 8.248209e-01 3.418900e+04 0.000000e+00

    logit_roc <- rocit(train$p_hat, train$over_50k)
    plot(logit_roc) # model seems pretty good at classification, hard to judge without a direct comparison.
    logit_roc$AUC #0.912

    ## [1] 0.9124105

    plot(logit_roc)$optimal # Youden = 0.657 Cutoff = 0.244

![](main_files/figure-markdown_strict/unnamed-chunk-11-1.png)

    ##     value       FPR       TPR    cutoff 
    ## 0.6572742 0.1977403 0.8550145 0.2441286

    # I'll use the cutoff of .244 informed by Youden's J statistic

    pred_probs <- predict.glm(main.effects.model, newdata = test_set, type = "response" )
    test_set <- test_set %>%
      mutate(over_50k_hat = ifelse(pred_probs > 0.2441286, 1, 0), # predicted 1 or 0
             p_hat = pred_probs) # predicted probability

    logit_meas <- measureit(test_set$p_hat, test_set$over_50k, measure = c("ACC", "SENS", "SPEC")) # accuracy, sensitivity, and specificity
    acc_table <- data.frame(Cutoff = 0.2441286, Sens = logit_meas$SENS, Spec = logit_meas$SPEC,  Acc = logit_meas$ACC)
    head(arrange(acc_table, desc(Acc)), n = 1) # 85.7% accuracy at Youden's cutoff

    ##      Cutoff      Sens      Spec       Acc
    ## 1 0.2441286 0.5861265 0.9398449 0.8570161

The logistic regression model’s accuracy = 85.7%. Since this is census
data and likely for a government client where regulations are a concern,
I prioritized interpretability over prediction. However, I’ll quickly
make a random forest model to get an idea of what a machine learning
algorithm can provide.

I won’t need to worry about separation concerns or binning variables, so
I’ll just use the original data. I’ll also use a 70/30 train/test split

    set.seed(12345)
    trainIndex <- createDataPartition(data$over_50k, p = 0.7, list = FALSE)
    # Split into train and validation_test
    train_rf <- data[trainIndex, ]
    test_rf <- data[-trainIndex, ]

    set.seed(12345)
    rf <- randomForest(factor(over_50k) ~ ., data = train_rf, ntree = 500) # create the model, no tuning

Random Forest metrics

    train_rf$p_hat <- predict(rf, type = "prob") # predicted probabilities
    train_rf <- train_rf %>% mutate(p_hat = p_hat[,"1"]) # the above line 
    # saves the p_hat variable as "p_hat[,"1"]" this line changes it to just "p_hat"

    somers2(train_rf$p_hat, train_rf$over_50k) # Concordance = 90.1%

    ##            C          Dxy            n      Missing 
    ## 9.009556e-01 8.019112e-01 3.419000e+04 0.000000e+00

    logit_roc <- rocit(train_rf$p_hat, train_rf$over_50k)
    logit_roc$AUC #0.901

    ## [1] 0.9009556

    plot(logit_roc) # model seems comparable to logistic regression
    plot(logit_roc)$optimal # Youden = 0.633 Cutoff = 0.269

![](main_files/figure-markdown_strict/unnamed-chunk-13-1.png)

    ##     value       FPR       TPR    cutoff 
    ## 0.6330633 0.1429783 0.7760417 0.2690355

    pred_probs <- predict(rf, newdata = test_rf, type = "prob")
    test_rf <- test_rf %>%
      mutate(over_50k_hat = ifelse(pred_probs[,2] > 0.2690355, 1, 0),# predicted 1 or 0
             p_hat = pred_probs[,2]) # predicted probability

    logit_meas <- measureit(test_rf$p_hat, test_rf$over_50k, measure = c("ACC", "SENS", "SPEC")) # accuracy, sensitivity, and specificity
    acc_table <- data.frame(Cutoff = 0.2690355, Sens = logit_meas$SENS, Spec = logit_meas$SPEC,  Acc = logit_meas$ACC)
    head(arrange(acc_table, desc(Acc)), n = 1) # accuracy at Youden cutoff= 86.27%

    ##      Cutoff      Sens      Spec       Acc
    ## 1 0.2690355 0.5974934 0.9437662 0.8626809

The random forest model (RF) performed better than the logistic
regression model (LR) in terms of accuracy. The RF’s concordance is ~
1.1% lower than the LR’s, and the maximum accuracy is ~0.6% higher at
the optimal cutoff. An interesting next step would be to tune the random
forest model and try other machine learning models like XGBoost to see
if the metrics can be further improved.
