// Hypergraph Explorer & Local Network Visualization
window.createNetwork = function(sourceId, targetId) {
    const container = document.getElementById('networkGraph');
    if (!container) return;
    
    d3.select('#networkGraph').html('');
    const width = container.clientWidth;
    const height = 400;

    const svg = d3.select('#networkGraph')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Mock local neighborhood nodes
    const nodes = [
        { id: sourceId, type: 'source' },
        { id: targetId, type: 'target' },
        { id: 'Neighbor A', type: 'relay' },
        { id: 'Neighbor B', type: 'relay' },
        { id: 'Neighbor C', type: 'relay' }
    ];

    const links = [
        { source: sourceId, target: 'Neighbor A' },
        { source: sourceId, target: 'Neighbor B' },
        { source: targetId, target: 'Neighbor B' },
        { source: targetId, target: 'Neighbor C' },
        { source: 'Neighbor A', target: 'Neighbor C' }
    ];

    // Hyperedge (group interaction) ellipse
    svg.append('ellipse')
        .attr('cx', width / 2)
        .attr('cy', height / 2)
        .attr('rx', 150)
        .attr('ry', 80)
        .attr('fill', 'rgba(139, 92, 246, 0.1)')
        .attr('stroke', '#8b5cf6')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5')
        .style('opacity', 0)
        .transition().duration(1000).style('opacity', 1);

    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2));

    const link = svg.selectAll('.link')
        .data(links)
        .enter().append('line')
        .attr('stroke', '#444')
        .attr('stroke-width', 2);

    const node = svg.selectAll('.node')
        .data(nodes)
        .enter().append('g')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));

    node.append('circle')
        .attr('r', d => (d.type === 'source' || d.type === 'target') ? 22 : 12)
        .attr('fill', d => {
            if (d.type === 'source') return '#38bdf8';
            if (d.type === 'target') return '#818cf8';
            return '#444';
        })
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

    node.append('text')
        .attr('dy', 30)
        .attr('text-anchor', 'middle')
        .attr('fill', '#cbd5e1')
        .style('font-size', '10px')
        .text(d => d.id);

    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node
            .attr('transform', d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }
    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }
    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }
};

// Initialize the main explorer if needed
document.addEventListener('DOMContentLoaded', () => {
    // (Main explorer logic can go here or be called separately)
});
