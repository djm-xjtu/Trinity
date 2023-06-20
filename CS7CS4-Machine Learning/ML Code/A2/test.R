library(HistData)
data(Minard.troops)
data(Minard.cities)
data(Minard.temp)
library(ggmap)
library(ggrepel)
library(grid)
library(gridExtra)


#instantiate get_stamenmap which is from the package ggmap
#initiate variables
value <- 7
ggmap_type <- "toner-hybrid"
leftValue <- 23.5
rightValue <- 40
topValue <- 56.3
bottomValue <- 53.4

#create a map using coordinates of Poland and Russia to plot the Minard map
poland_russia_Map <- get_stamenmap((c(left = leftValue, bottom = bottomValue, right = rightValue, top = topValue)),
zoom=value, maptype = ggmap_type)%>% ggmap()

#Misc variables
joinType <- "round"
cityLabelColor <- "#990000"
fontSize <- 15

#join the Napolean geo-cords with the map
napoleonCoords <- poland_russia_Map +
geom_path( lineend = joinType,data=Minard.troops, aes(x = long, y = lat, group = group, color = direction, size = survivors))

#adding geo-points
napoleonCityCoords <- napoleonCoords + geom_point(data = Minard.cities, aes(x = long, y = lat),
		 color = cityLabelColor)

#city labels
napoleonCityText <- napoleonCityCoords + geom_text_repel(data = Minard.cities, aes(x = long, y = lat, label = city),
			  color = cityLabelColor, fontface =fontSize)

rangeMin <- 0.5
rangeMax <- 10

adavancingColor <- "#996633"
retreatingColor <- "black"

#Width range
widthSize <- napoleonCityText + scale_size(range = c(rangeMin, rangeMax))

finalRoute <- widthSize + scale_colour_manual(values = c(adavancingColor, retreatingColor)) +
theme(panel.border=element_rect(fill=NA, size = 1))

finalRoute
#Temperature mapping
minardTempSpot<-ggplot(data = Minard.temp, aes(x = long, y = temp))
#Draw a temperature line along with Napolean map
drawTempLine <- minardTempSpot + geom_line()
#DisplayText
finalTempPlot <- drawTempLine + geom_text(aes(label = temp))

#new gird to join both the plots
grid::grid.newpage()
g1 <- ggplotGrob(finalRoute)
g2 <- ggplotGrob(finalTempPlot)
colnames(g1) <- paste0(seq_len(ncol(g1)))
colnames(g2) <- paste0(seq_len(ncol(g2)))

grid.draw(gtable_combine(g1, g2, along=2))