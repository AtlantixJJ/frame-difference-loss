var data;

function generateHtmlTable(data, container) {
    var html = '<table  class="table table-condensed table-hover table-striped">';

    if(typeof(data[0]) === 'undefined') {
        return null;
    } else {
        $.each(data, function( index, row ) {
            //bind header
            if(index == 0) {
                html += '<thead>';
                html += '<tr>';
                $.each(row, function( index, colData ) {
                    html += '<th>';
                    html += colData;
                    html += '</th>';
                });
                html += '</tr>';
                html += '</thead>';
                html += '<tbody>';
            } else {
                html += '<tr>';
                $.each(row, function( index, colData ) {
                    html += '<td>';
                    html += colData;
                    html += '</td>';
                });
                html += '</tr>';
            }
        });
        html += '</tbody>';
        html += '</table>';
        $(container).append(html);
    }
}

function showcsv(url, container) {
    $.ajax({
        type: "GET",  
        url: url,
        dataType: "text",       
        success: function(response)  
        {
            data = $.csv.toArrays(response);
            generateHtmlTable(data, container);
        }
    });
}