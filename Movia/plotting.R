library(readr)
library(data.table)
library(rgdal)
library(ggplot2)
library(ggthemes)
library(scales)
library(leaflet)
library(sp)
library(sf)

read_results <- function(file) {
  results <- as.data.table(read_csv(file))
  results$Error = results$LinkTravelTime - results$LinkTravelTime_Predicted
  results$Hour <- as.numeric(format(results$DateTime, "%H"))
  results$DayType = factor(results$DayType)
  results$LinkName <- factor(results$LinkName, levels = unique(results$LinkName[order(results$LineDirectionLinkOrder)]))
  results$LinkOrder <- as.integer(results$LinkName)
  results
}

plot_result_errors <- function(ds = list(), labels = list()) {
  p <-  ggplot()
  
  for (i in seq_len(min(length(ds), length(labels)))) {
    loop_input = paste("stat_ecdf(data = ds[[i]], aes(x = Error, colour = '",labels[[i]],"'))", sep="")
    p <- p + eval(parse(text=loop_input)) 
  }
  
  p <- p +
    scale_y_continuous(labels=percent) + 
    facet_grid(LinkOrder ~ .) +
    theme_tufte() +
    theme(panel.grid = element_line(size = .25, linetype = "solid", color = "black")) +
    theme(legend.position = "bottom")
  
  p
}

results_lr_single <- read_results('../data/results_lr_single.csv')
results_lr_multiple <- read_results('../data/results_lr_multiple.csv')

plot_result_errors(list(results_lr_single, results_lr_multiple), list('LR single', 'LR multiple')) +
  xlim(-100, 150) +
  theme(axis.text.x = element_text(size=7)) +
  theme(axis.text.y = element_text(size=7)) +
  ggsave('plots/results_lr_errors.pdf', width = 210, height = 148, units = "mm")

results_svr_single <- read_results('../data/results_svr_single.csv')
results_svr_multiple <- read_results('../data/results_svr_multiple.csv')
