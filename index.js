var express = require('express');
var app = express();
var sleep=require('sleep');
var bodyParser = require('body-parser')
const ogs = require('open-graph-scraper');
var fs=require('fs');
const { SSL_OP_EPHEMERAL_RSA } = require('constants');
let outputa="";
app.use(bodyParser.urlencoded({ extended: true })); 
app.get('/', function (req, res) {
    res.write('<html><body>');
    res.write("Enter URL<br>");
    res.write("<form action='/search' method='POST'>");
    res.write('<input type="url" name="geturl" required><br>');
    res.write('<input type="submit" name="submit">');
    res.write('</form></body></html>')
    res.end();
})
app.post('/search',function(req,res){
  

 
   
  const options = { url: req.body.geturl };
  ogs(options, (error, results, response) => {
     //writing the JSON object to STring using stringify
     fs.writeFile('abc.txt',JSON.stringify(results),function(err){
            if(err) throw err;
     })
     outputa=JSON.stringify(results);
     console.log(results);
    console.log('error:', error); // This is returns true or false. True if there was a error. The error it self is inside the results object.
   // console.log(JSON.stringify(results)); // This contains all of the Open Graph results
   
   
  })
  sleep.sleep(7);
// waiting to get data from different source
  res.redirect('/result');
 
 
})
app.get('/result',function(req,res){
fs.readFile('abc.txt','utf8',(err,data)=>{
   if(err){
      console.error(err);
      return
   }
   console.log(outputa);
   res.write("<html><meta http=equiv='refresh' content='5; url=/result'><body>");//refreshing page every 5 seconds
   res.write(outputa);
   res.write("<br><br><h1><a href='/'>click here to go back</a></h1>")
   //res.write(data); this is for reading data from file
   res.end();
})
})

var server = app.listen(process.env.PORT ||3000, function () {
   var host = server.address().address
   var port = server.address().port
   
   console.log("Example app listening at http://%s:%s", host, port)
})