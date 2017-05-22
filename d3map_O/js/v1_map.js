"use strict";

/*
* initial source: https://bost.ocks.org/mike/leaflet/
* this map uses d3 for geoJSON and some overlays
*/

// leaflet map tiles
var mbAttr = '&copy; <a href="https://www.mapbox.com/map-feedback/">Mapbox</a> &copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
L.mapbox.accessToken = 'pk.eyJ1IjoibXJrYWlrZXYiLCJhIjoiY2luZGF4NzA2MDA1Z3d6bHlwbWZ4YWI4YiJ9.5tLR_2fjmu95FYEaEAljYw';
var street = L.tileLayer('https://api.mapbox.com/v4/mapbox.streets/{z}/{x}/{y}.png?access_token=' + L.mapbox.accessToken, {
	attribution: mbAttr
});

var grayscale = L.tileLayer('https://api.mapbox.com/v4/mapbox.light/{z}/{x}/{y}.png?access_token=' + L.mapbox.accessToken, {
	attribution: mbAttr
});

var terrain = L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
	attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
});

var googlesat = L.tileLayer('https://mt1.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}', {
	attribution: "Map data &copy;2016 Google"
});

var positron = L.tileLayer('http://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png', {
	attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="http://cartodb.com/attributions">CartoDB</a>'
});

var map1 = L.map('map1', {
	center: [40.711510, -73.935242],
	zoom: 11,
	layers: [street]
});

// control map layers
var baseLayers = {
	"Streets (Mapbox)": street,
	"Grayscale (Mapbox)": grayscale,
	"Grayscale (CartoDB)": positron,
	"Terrain (OSM)": terrain,
	"Satellite (Google)": googlesat
};

var d3Layer = L.Class.extend({
	initialize: function() {
		return;
	},
	onAdd: function() {
		d3.select("div#map1 .legend").style("display", "block");
		d3.select("div#map1 .regions").style("display", "block");
	},
	onRemove: function() {
		d3.select("div#map1 .regions").style("display", "none");
		d3.select("div#map1 .legend").style("display", "none");
	},
});

var svgLayer = new d3Layer();

var overlays = {
	"GeoJSON Regions": svgLayer
};
L.control.layers(baseLayers, overlays).addTo(map1);

// d3 map data
var svgMap = d3.select(map1.getPanes().overlayPane).append("svg"),
	g = svgMap.append("g").attr("class", "leaflet-zoom-hide regions"),
	g_circles = svgMap.append("g").attr("class", "leaflet-zoom-hide");

// Define the div for the tooltip
var div = d3.select("body").append("div")	
    .attr("class", "tooltip")
    .style("opacity", 0);

 // Setting color domains(intervals of values) for our map

var flag = 0
var color_domain = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]

var color = d3.scale.threshold()
  .domain(color_domain)
  .range(colorbrewer.YlOrRd[9])

d3.json("data/taxi_zones/taxi_zones.geojson", function(error, collection) {

	if (error) throw error;

	var transform = d3.geo.transform({
			point: projectPoint
	}),
	path = d3.geo.path().projection(transform);

	var feature = g.selectAll("path")
		.data(collection.features)
		.enter().append("path");

	g.selectAll('text')
		.data(collection.features)
		.enter()
		.append("text")
		.text(function(d,i){
			return (i+1)
		})
		.attr("x", function(d){
        return path.centroid(d)[0];
    })
    .attr("y", function(d){
        return  path.centroid(d)[1];
    })
    .attr("text-anchor","middle")
    .attr('font-size','9pt')
    .attr('font-weight', 'bold');

	// Handle zoom of the map and repositioning of d3 overlay
	map1.on("viewreset", reset);
	reset();

	function getCheckedOrigin(){

		var origins = document.getElementsByName('origin');
		for(var i = 0; i < origins.length; i++){
		    if(origins[i].checked){
		       	return origins[i].value;
		    }
		}
	}

	function getCheckedDestination(){

		var destinations = document.getElementsByName('destination')
		for(var i=0; i<destinations.length; i++){
			if(destinations[i].checked){
				return destinations[i].value;
			}
		}
	}

	function parseData(data, IDor, IDdes){

		var corrById = {};
		var numberPattern = /\d+/g;	
		var parsed_data = data[IDor-1][IDdes-1].split(',')

		for(var i in parsed_data){
			corrById[parsed_data[i].split(':')[0].match(numberPattern)[0]] = parseFloat(parsed_data[i].split(':')[1])
		}
		return corrById
	}

	function emphasizeRegion(){

		var orID = getCheckedOrigin();
		var desID = getCheckedDestination();

		d3.select(feature[0][parseInt(orID)-1])
			.attr("d", path)
			.style("fill", "#8FBC8F")
			.style("stroke", "black")
			.style("stroke-width", 2)

		d3.select(feature[0][parseInt(desID)-1])
			.attr("d", path)
			.style("fill", "#87CEFA")
			.style("stroke", "black")
			.style("stroke-width", 2)
	}

	function resetRegion(){

		//go back to original features 
		feature.attr("d", path)
			.style("stroke", 'black')
      .style("fill", "none")
      .style("stroke-width", 2)
	}

	function drawHeatMap(){

		d3.csv("data/CorrMat.csv", function(error, data){

				var orID = getCheckedOrigin()
				var destID = getCheckedDestination()
				var corrById = parseData(data, orID, destID)

				feature.attr("d", path)
	   				.style("fill", function(d, i){
	   					if ((i != parseInt(orID)-1) && (i!= parseInt(destID)-1)){
	   						return color(corrById[i+1]);
	   					}
	   					if (i == parseInt(orID-1)){
	   						return "#8FBC8F"
	   					}

	   					else{
	   						return "#87CEFA"
	   					}
	  				})
					.style("opacity", .9)
					.on("mouseover", function(d, i) {	
						d3.select(this)
							.transition()
							.duration(500)
							.style("opacity", .55)	
            			div.transition()		
                			.duration(200)		
                			.style("opacity", .9);		
            			div	.html(
            				"<strong> Zone:</strong><span style='color:#8B0000'>" + (i+1)  + "</span>\n" + 
            				"<strong>Correlation:</strong> <span style='color:#8B0000'>" 
            				+ parseFloat(corrById[i+1]).toFixed(4) + "</span>")	
                			.style("left", (d3.event.pageX + 16) + "px")		
                			.style("top", (d3.event.pageY + 16) + "px")})
          .on("mouseout", function(d) {	
            d3.select(this)
							.transition()
							.duration(500)
							.style("opacity", .9)	
            			div.transition()		
                			.duration(500)		
                			.style("opacity", 0);				   
				})
		})
	}

	function reset(){
		
		var bounds = 
				path.bounds(collection),
					topLeft = bounds[0],
					bottomRight = bounds[1];
				svgMap.attr("width", bottomRight[0] - topLeft[0])
					.attr("height", bottomRight[1] - topLeft[1])
					.style("left", topLeft[0] + "px")
					.style("top", topLeft[1] + "px");
				g.attr("transform", "translate(" + -topLeft[0] + "," + -topLeft[1] + ")");
				g_circles.attr("transform", "translate(" + -topLeft[0] + "," + -topLeft[1] + ")");

		// Add colors and other fillings for every feature
		feature.attr("d", path)
			.style("stroke", 'black')
       		.style("fill", "none")
       		.style("stroke-width", 2)

    if (flag!=0){
    	g.selectAll("text")
				.text("");
    }

    flag = flag + 1;

    //keep having the emphasized region while zooming
    var orID = getCheckedOrigin();
		var desID = getCheckedDestination();

		d3.select(feature[0][parseInt(orID)-1])
			.attr("d", path)
			.style("fill", "gray")
			.style("stroke", "black")
			.style("stroke-width", 2)

		d3.select(feature[0][parseInt(desID)-1])
			.attr("d", path)
			.style("fill", "gray")
    	.style("stroke", "black")
    	.style("stroke-width", 2)

    drawHeatMap();
	}

	d3.selectAll("#ods")
		.on("change", function(){
			resetRegion()
			emphasizeRegion()
			drawHeatMap()
		})
	
	// Use Leaflet to implement a D3 geometric transformation.
	function projectPoint(x, y) {
		var point = map1.latLngToLayerPoint(new L.LatLng(y, x));
		this.stream.point(point.x, point.y);
	}
});
