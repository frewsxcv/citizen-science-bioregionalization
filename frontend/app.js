const map = new maplibregl.Map({
    container: 'map',
    style: 'https://api.maptiler.com/maps/streets/style.json?key=get_your_own_OpIi9ZULNHzrESv6T2vL', // Replace with your MapTiler key
    center: [-20, 65],
    zoom: 4
});

map.on('load', () => {
    fetch('aggregations.json')
        .then(response => response.json())
        .then(data => {
            const geojson = {
                type: 'FeatureCollection',
                features: data.map(cluster => ({
                    type: 'Feature',
                    geometry: cluster.boundary,
                    properties: {
                        cluster: cluster.cluster,
                        significant_taxa: cluster.significant_taxa,
                        color: cluster.color,
                        darkened_color: cluster.darkened_color
                    }
                }))
            };

            map.addSource('clusters', {
                type: 'geojson',
                data: geojson
            });

            map.addLayer({
                id: 'clusters-fill',
                type: 'fill',
                source: 'clusters',
                paint: {
                    'fill-color': ['get', 'color'],
                    'fill-opacity': 0.7
                }
            });

            map.addLayer({
                id: 'clusters-outline',
                type: 'line',
                source: 'clusters',
                paint: {
                    'line-color': ['get', 'darkened_color'],
                    'line-width': 2
                }
            });

            map.on('click', 'clusters-fill', (e) => {
                const properties = e.features[0].properties;
                const clusterId = properties.cluster;
                const significantTaxa = JSON.parse(properties.significant_taxa);

                const detailsDiv = document.getElementById('cluster-details');
                let html = `<h3>Cluster ${clusterId}</h3>`;

                if (significantTaxa.length > 0) {
                    significantTaxa.sort((a, b) => b.cluster_count - a.cluster_count);
                    html += '<ul>';
                    significantTaxa.forEach(taxa => {
                        html += `<li>
                            ${taxa.image_url ? `<img src="${taxa.image_url}?width=50" alt="${taxa.scientific_name}" style="width:50px; height:50px; object-fit: cover; margin-right: 10px; float: left;">` : '<div style="width: 50px; height: 50px; float: left; margin-right: 10px; background: #eee;"></div>'}
                            ${taxa.scientific_name}<br>
                            <small>
                                p=${taxa.p_value.toExponential(2)},
                                log2fc=${taxa.log2_fold_change.toFixed(2)},
                                in-cluster=${taxa.cluster_count},
                                neighbors=${taxa.neighbor_count}
                            </small>
                            <div style="clear: both;"></div>
                        </li>`;
                    });
                    html += '</ul>';
                } else {
                    html += '<p>No significant taxa for this cluster.</p>';
                }

                detailsDiv.innerHTML = html;
            });

            map.on('mouseenter', 'clusters-fill', () => {
                map.getCanvas().style.cursor = 'pointer';
            });

            map.on('mouseleave', 'clusters-fill', () => {
                map.getCanvas().style.cursor = '';
            });
        });
});