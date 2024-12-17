library(dplyr)
library(astsa)
library(forecast)
library(lubridate)
library(ggcorrplot)



SeoulBikeData <- read.csv("SeoulBikeData.csv", header=TRUE, 
                          col.names=c("Date", "Rented_Bike_Count", "Hour", 
                                      "Temperature", "Humidity", "Wind_speed", 
                                      "Visibility", "Dew_point_temperature", 
                                      "Solar_Radiation", "Rainfall", "Snowfall",
                                      "Seasons", "Holiday", "Functioning_Day"))
SeoulBikeData <- as.data.frame(SeoulBikeData)

SeoulBikeData$datetime <- dmy(SeoulBikeData$Date) + hours(SeoulBikeData$Hour)

SeoulBikeData$Temp <- SeoulBikeData$Temperature - mean(SeoulBikeData$Temperature)
SeoulBikeData$Temp2 <- (SeoulBikeData$Temp)^2

morning = SeoulBikeData %>% filter(Hour == 8)
evening = SeoulBikeData %>% filter(Hour == 18)

morning_train <- morning[1:(nrow(morning) - 7), ]
morning_test <- morning[(nrow(morning) - 7 + 1):nrow(morning), ]
evening_train <- evening[1:(nrow(evening) - 7), ]
evening_test <- evening[(nrow(evening) - 7 + 1):nrow(evening), ]



# Correlation Matrix
cor_matrix <-
  ggcorrplot(cor(SeoulBikeData[, c(2, 4:11)]), 
             method = "square", 
             lab = TRUE,          
             colors = c("blue", "white", "red"), 
             ggtheme = theme_minimal(),
             digits = 1)  
ggsave("CorrMatrix.png", plot = cor_matrix, width = 10, height = 6)



# use *morning_train* as training set.
ts_morning_train <- ts(morning_train["Rented_Bike_Count"])[, 1]
# png("ts_plot_diff.png", width = 5000, height = 1000, res = 300)
par(mar = c(4, 4, 2, 1))
plot(ts_morning_train, main="TS of Rented Bike Count (morning_train)")
# dev.off()

acf2(ts_morning_train, main="ACF/PACF plots for morning data")

lm_morning_train_all <- lm(Rented_Bike_Count~ 
                             Temperature +
                             Humidity + 
                             Wind_speed + 
                             Visibility + 
                             Dew_point_temperature + 
                             Solar_Radiation + 
                             Rainfall + 
                             Snowfall,
                           data=morning_train)
summary(lm_morning_train_all)

lm_morning_train <- lm(Rented_Bike_Count ~
                         Solar_Radiation +
                         Rainfall,
                       data=morning_train)
summary(lm_morning_train)

# set up explanatory variables
xreg <- cbind(morning_train$Solar_Radiation,
              morning_train$Rainfall)

#png("ts_plot.png", width = 5000, height = 1000, res = 300)
par(mar = c(4, 4, 2, 1))
plot(diff(ts(resid(lm_morning_train),frequency=7),1),main="TS plot of residual")
#dev.off()

acf2(ts(resid(lm_morning_train), frequency=7), main="ACF/PACF plots of residuals (morning)")

# try SARIMA(0,1,1)x(0,1,1)_7
morning_sarima_111010 <- sarima(ts(morning_train$Rented_Bike_Count), p=0, d=1, q=1, P=0, D=1, Q=1, S=7, 
                                xreg = xreg)

# forecast
future_xreg <- cbind(morning_test$Solar_Radiation,
                     morning_test$Rainfall)

# use SARIMA(0,1,1)x(0,1,1)_7
week_forecast <- sarima.for(ts(morning_train$Rented_Bike_Count), 
                            p=0, d=1, q=1, P=0, D=1, Q=1, S=7,
                            xreg=xreg,
                            n.ahead=7,
                            newxreg=future_xreg)




# use *evening_train* as training set.

ts_evening_train <- ts(evening_train["Rented_Bike_Count"])[, 1]
# png("ts_plot_diff.png", width = 5000, height = 1000, res = 300)
par(mar = c(4, 4, 2, 1))
plot(ts_evening_train, main="TS of Rented Bike Count (evening_train)")
# dev.off()

acf2(ts_evening_train)

lm_evening_train_all <- lm(Rented_Bike_Count~ 
                             Temperature +
                             Humidity + 
                             Wind_speed + 
                             Visibility + 
                             Dew_point_temperature + 
                             Solar_Radiation + 
                             Rainfall + 
                             Snowfall,
                           data=evening_train)
summary(lm_evening_train_all)

# use Humidity and Dew_point_temperature
lm_evening_train <- lm(Rented_Bike_Count~ 
                         Humidity + 
                         Dew_point_temperature,
                       data=evening_train)
summary(lm_evening_train)

# set up explanatory variables
xreg_evening <- cbind(evening_train$Humidity,
                      evening_train$Dew_point_temperature)

#png("ts_plot.png", width = 5000, height = 1000, res = 300)
par(mar = c(4, 4, 2, 1))
plot(ts(resid(lm_evening_train), frequency=7),main="TS plot of residual")
#dev.off()

acf2(ts(resid(lm_evening_train), frequency=7),main="ACF/PACF plots of residuals (evening)")

# try SARIMA(2,1,5)x(2,1,2)_7
evening__sarima_215212 <- sarima(ts(evening_train$Rented_Bike_Count), p=2, d=1, q=5, P=2, D=1, Q=2, S=7, 
                                 xreg = xreg_evening)

# forecast
future_xreg_evening <- cbind(evening_test$Humidity,
                             evening_test$Dew_point_temperature)

# use SARIMA(2,1,5)x(2,1,2)_7
week_forecast_evening <- sarima.for(ts(evening_train$Rented_Bike_Count), 
                                    p=2, d=1, q=5, P=2, D=1, Q=2, S=7,
                                    xreg=xreg_evening,
                                    n.ahead=7,
                                    newxreg=future_xreg_evening)
