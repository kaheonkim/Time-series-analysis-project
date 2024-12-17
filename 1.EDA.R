library(astsa)
library(dplyr)
library(forecast)
SeoulBikeData <- read.csv("SeoulBikeData.csv", header=TRUE, 
                          col.names=c("Date", "Rented_Bike_Count", "Hour", 
                                      "Temperature", "Humidity", "Wind_speed", 
                                      "Visibility", "Dew_point_temperature", 
                                      "Solar_Radiation", "Rainfall", "Snowfall",
                                      "Seasons", "Holiday", "Functioning_Day"))

# Load necessary libraries
library(ggplot2)

# Assuming you have already read the data (as shown in your code)
# Plot Rented_Bike_Count against Datetime (Date + Hour)
plot(SeoulBikeData_100days$Datetime, SeoulBikeData_100days$Rented_Bike_Count, 
     type = 'l',  # Type 'l' for line plot
     main = "Count of bike rental", 
     xlab = "Date", 
     ylab = "Bike Rental")

axis.Date(1, at = seq(min(SeoulBikeData_100days$Datetime), max(SeoulBikeData_100days$Datetime), by = "12 hours"),
          format = "%Y-%m-%d %H:%M")
acf2(ts(SeoulBikeData["Rented_Bike_Count"]), main = 'ACF/PACF plots for original data (hourly)')

boxplot(Rented_Bike_Count ~ Hour, 
        data = SeoulBikeData,
        main = "Boxplot of Bike Rentals by Hour",
        xlab = "Hour",
        ylab = "Bike Rental",
        col = rainbow(24), # Use a color for each hour
        las = 1)   

morning = SeoulBikeData %>%filter(Hour == 8)
rent_morning = ts(morning["Rented_Bike_Count"])
evening = SeoulBikeData%>%filter(Hour == 18)
rent_evening = ts(evening["Rented_Bike_Count"])



