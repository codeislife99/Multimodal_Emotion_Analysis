
$(document).ready(function(){

   var count=1;
   var player=document.getElementById('myVideo');
   var mp4Vid = document.getElementById('mp4Source');
   player.addEventListener('ended',myHandler,false);
     
   $(mp4Vid).attr('src', "videos/video"+count+".mp4");
      
  
    $.getJSON( "predictions/json1.json", function( res ) {
      console.log("yadayada");
      
      console.log(res);
      var data = [
                {
                x: ['Content/Relief/Love', 'Happiness/Enthusiastic', 'Anger/Disgust/Fear', 'Sadness/Emptiness', 'Neutral'],                
                y: [res.prediction[0], res.prediction[1],res.prediction[2], res.prediction[3], res.prediction[4]],
                type: 'bar'
              }
            ]; 
      var data_vv = [
                {
                x: ['Content/Relief/Love', 'Happiness/Enthusiastic', 'Anger/Disgust/Fear', 'Sadness/Emptiness', 'Neutral'],                
                y: [res.vv[0], res.vv[1],res.vv[2], res.vv[3], res.vv[4]],
                type: 'bar'
              }
            ]; 
      var data_text = [
                {
                x: ['Content/Relief/Love', 'Happiness/Enthusiastic', 'Anger/Disgust/Fear', 'Sadness/Emptiness', 'Neutral'],                
                y: [res.text[0], res.text[1],res.text[2], res.text[3], res.text[4]],
                type: 'bar'
              }
            ];      
    var layout = {
      yaxis: {range: [0, 1]}
    };
      player.load();
      player.play();

    Plotly.newPlot('hi', data, layout);
    Plotly.newPlot('vv', data_vv, layout);
    Plotly.newPlot('text', data_text, layout);
    $("#hi_emo").html("Prominent Emotion : "+res.max);
    $("#hi_acc").html("Accuracy : "+res.acc);
    $("#transcript").html(res.transcript);
    $("#vv_emo").html("Prominent Emotion : "+res.vv_max);
    $("#vv_acc").html("Accuracy : "+res.vv_acc);
    $("#text_emo").html("Prominent Emotion : "+res.text_max);
    $("#text_acc").html("Accuracy : "+res.text_acc);
    $("#actual").html("Actual Emotion : "+res.actual);
  });


   function myHandler(e)
   {

      if(!e) 
      {
         e = window.event; 
      }
      count++;
      $(mp4Vid).attr('src', "videos/video"+count+".mp4");
      //player.load();
      //player.play();

      
    $.getJSON( "predictions/json"+count+".json", function( res ) {
      console.log("yadayada");
      
      console.log(res);
      var data = [
                {
                x: ['Content/Relief/Love', 'Happiness/Enthusiastic', 'Anger/Disgust/Fear', 'Sadness/Emptiness', 'Neutral'],                
                y: [res.prediction[0], res.prediction[1],res.prediction[2], res.prediction[3], res.prediction[4]],
                type: 'bar'
              }
            ];

      var data_vv = [
                {
                x: ['Content/Relief/Love', 'Happiness/Enthusiastic', 'Anger/Disgust/Fear', 'Sadness/Emptiness', 'Neutral'],                
                y: [res.vv[0], res.vv[1],res.vv[2], res.vv[3], res.vv[4]],
                type: 'bar'
              }
            ]; 
      var data_text = [
                {
                x: ['Content/Relief/Love', 'Happiness/Enthusiastic', 'Anger/Disgust/Fear', 'Sadness/Emptiness', 'Neutral'],                
                y: [res.text[0], res.text[1],res.text[2], res.text[3], res.text[4]],
                type: 'bar'
              }
            ]; 
    var layout = {
      yaxis: {range: [0, 1]}
    };
      player.load();
      player.play();

          Plotly.newPlot('hi', data, layout);
          Plotly.newPlot('vv', data_vv, layout);
          Plotly.newPlot('text', data_text, layout);
          $("#hi_emo").html("Prominent Emotion : "+res.max);
          $("#hi_acc").html("Accuracy : "+res.acc);
          $("#transcript").html(res.transcript);
          $("#vv_emo").html("Prominent Emotion : "+res.vv_max);
          $("#vv_acc").html("Accuracy : "+res.vv_acc);
          $("#text_emo").html("Prominent Emotion : "+res.text_max);
          $("#text_acc").html("Accuracy : "+res.text_acc);
          $("#actual").html("Actual Emotion : "+res.actual);
      });


   }
                         




});

