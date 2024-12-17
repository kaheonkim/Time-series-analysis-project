
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

mae <- function(actual, predicted) {
  (mean(abs(actual - predicted)))
}

morning = SeoulBikeData %>%filter(Hour == 8)
rent_morning = ts(morning["Rented_Bike_Count"])
evening = SeoulBikeData%>%filter(Hour == 18)
rent_evening = ts(evening["Rented_Bike_Count"])

rent_morning_week = ts(rent_morning[1:(length(rent_morning)-7),], freq = 7)
rent_morning_week_label = ts(rent_morning[(length(rent_morning)-7+1):length(rent_morning),])
rent_evening_week = ts(rent_evening[1:(length(rent_evening)-7),], freq = 7)
rent_evening_week_label = ts(rent_evening[(length(rent_evening)-7+1):length(rent_evening),])
rent_morning_month = ts(rent_morning[1:(length(rent_morning)-30),], freq = 7)
rent_morning_month_label = ts(rent_morning[(length(rent_morning)-30+1):length(rent_morning),])
rent_evening_month = ts(rent_evening[1:(length(rent_evening)-30),], freq = 7)
rent_evening_month_label = ts(rent_evening[(length(rent_evening)-30+1):length(rent_evening),])

# Weekly
plot(rent_morning_week, type = 'l')
acf2(rent_morning_week)

model_morning <- auto.arima(rent_morning_week, D = 1, seasonal = TRUE)
summary(model_morning)
sarima(rent_morning_week, 1, 0, 1, 0, 1, 1, 7)

plot(rent_evening_week, type = 'l')
acf2(rent_evening_week)
model_evening <- auto.arima(rent_evening_week, D = 1, seasonal = TRUE)
summary(model_evening)
sarima(rent_evening_week,3, 1, 1, 0, 1, 1, 7)
pred_morning_week = sarima.for(rent_morning_week,7,1, 0, 1, 0, 1, 1, 7)
print(rmse(c(pred_morning_week$pred), c(rent_morning_week_label)))
print(mae(c(pred_morning_week$pred), c(rent_morning_week_label)))
pred_evening_week = sarima.for(rent_evening_week, 7, 3, 1, 1, 0, 1, 1, 7)
print(rmse(c(pred_evening_week$pred), c(rent_evening_week_label)))
print(mae(c(pred_evening_week$pred), c(rent_evening_week_label)))





# Monthly

plot(rent_morning_month, type = 'l')
acf2(rent_morning_month)

model_morning <- auto.arima(rent_morning_month, D = 1, seasonal = TRUE)
summary(model_morning)
sarima(rent_morning_month, 1, 0, 1, 0, 1, 1, 7)
plot(rent_evening_month, type = 'l')
acf2(rent_evening_month)
model_evening <- auto.arima(rent_evening_month, D = 1, seasonal = TRUE)
summary(model_evening)
sarima(rent_evening_month, 5, 1, 1, 1, 1, 1, 7)
model_evening <- auto.arima(rent_morning_month, D = 1, seasonal = TRUE)
summary(model_evening)
pred_morning_month = sarima.for(rent_morning_month, 30,1, 0, 1, 0, 1, 1, 7)
print(rmse(c(pred_morning_month$pred), c(rent_morning_month_label)))
print(mae(c(pred_morning_month$pred), c(rent_morning_month_label)))
pred_evening_month = sarima.for(rent_evening_month, 30, 5, 1, 1, 1, 1, 5, 7)
print(rmse(c(pred_evening_month$pred), c(rent_evening_month_label)))
print(mae(c(pred_evening_month$pred), c(rent_evening_month_label)))

