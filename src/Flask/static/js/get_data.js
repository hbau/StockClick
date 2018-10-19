$( "#target").click(function() {
             var data = {};
             data.country = $("#country").val();
             data.period = $("#period").val();
             data.date = $("#date").val();
             $.ajax({
                type : "POST",
                url : "/score",
                data: JSON.stringify(data, null, '\t'),
                contentType: 'application/json;charset=UTF-8',
                success: function(result) {
                  if(result=="missing"){
                    window.alert("Data Missing for specified country for the given period!");
                  } else {
                  // window.alert(result);
                  display_data(result);
                  }
                }
              });
             })

function display_data(str) {
  document.getElementById("demo").innerHTML = "";
  str = str.replace(/\'/g, '\"');
  var result = JSON.parse(str);
  for (event in result) {
    value ="";
      if(result[event].avg_score < 0) {
        value = '<button class="btn btn-danger" type="button"><b>'+event+'</b>  Mentions '+'<span class="badge">'+result[event].Mentions+'</span>'+' Goldstein Scale '+'<span class="badge">'+result[event].avg_score+'</span>'+' Article Tone '+'<span class="badge">'+result[event].avg_tone+'</span></button>'
      } else if (result[event].avg_score > 0) {
        value = '<button class="btn btn-success" type="button"><b>'+event+'</b>  Mentions '+'<span class="badge">'+result[event].Mentions+'</span>'+' Goldstein Scale '+'<span class="badge">'+result[event].avg_score+'</span>'+' Article Tone '+'<span class="badge">'+result[event].avg_tone+'</span></button>'
      } else {
        value = '<button class="btn btn-info" type="button"><b>'+event+'</b>  Mentions '+'<span class="badge">'+result[event].Mentions+'</span>'+' Goldstein Scale '+'<span class="badge">'+result[event].avg_score+'</span>'+' Article Tone '+'<span class="badge">'+result[event].avg_tone+'</span></button>'
      }
      document.getElementById("demo").innerHTML += value +"<br/><br/>";
    }
  }
  // var final = JSON.stringify(obj)
