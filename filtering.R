library(tidyverse)
library(lubridate)
library(sf)

Sys.setlocale("LC_TIME", "C")

# Load the dataset.
df <- read_tsv('/data/users_data/douglas/src/metrics/data/locations_wifi.tsv', col_names = FALSE)

# The original dataset (exported from Macaco DB) has all these columns.
colnames(df) <- c("measurement_id", 
                  "device_id", 
                  "timestamp_fire", 
                  "timestamp", 
                  "provider", 
                  "accuracy", 
                  "lat", 
                  "lon", 
                  "wifi_timestamp", 
                  "wifi_name", 
                  "mac_address")

# Remove rows with NAs.
df <- df %>%
    na.omit()

# Convert timestamp and timestamp_fire to date format.
df <- df %>%
    mutate(timestamp = as_datetime(timestamp / 1000),
          timestamp_fire = as_datetime(timestamp_fire / 1000))

# Remove entries that are older than 2015.
df <- df %>%
    filter(timestamp_fire >= '2015-01-01')

# Throw out all entries that have timestamp > timestamp_fire + 5min. 
# A new "get position" event is triggered every 5 min. 
# In order to get one geographical position at each 5min interval, 
# we keep only the most accurate location received before timestamp > timestamp_fire + 5min.
df <- df %>%
    filter(timestamp > timestamp_fire & timestamp < timestamp_fire + minutes(5))

# Throw out GPS entries that have timestamp > timestamp_fire + 30s. 
# We set a threshold of 30s to get the answers from the OS after the event "get position " is triggered,
# which keeps sending more accurate locations of users.
df <- df %>%
    filter(!(provider == "gps" & timestamp > timestamp_fire + seconds(30)))

# For each user, and for each measurement_ID of a user, select the location with larger accuracy.
df <- df %>%
    group_by(device_id, measurement_id) %>%
    top_n(-1, accuracy) %>%
    distinct(accuracy, .keep_all = TRUE) %>%
    ungroup()

# Convert lat/lon to decimal degrees.
df <- df %>%
    mutate(lat = lat / 1000000, lon = lon / 1000000)

# Number of users in the dataset.
length(unique(df$device_id))

# Write the full dataset to a file.
# This is the dataset that will be read by the data processing scripts.
system.time({
  write_tsv(df, "/data/users_data/douglas/src/metrics/data/macaco_all.tsv")
})


