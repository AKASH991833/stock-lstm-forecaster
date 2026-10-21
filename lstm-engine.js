/* stock-lstm-forecaster — small LSTM (TF.js, CPU backend) vs naive baseline on
   windowed log-returns. Synthetic GBM generator + built-in sample series.
   HONEST: backtested on a holdout; markets are near-random-walk, and the app says so. */
(function(){
'use strict';
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(rnd){let u=0,v=0;while(!u)u=rnd();while(!v)v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
// synthetic stock: GBM with regime switches + volatility clustering
function synthStock(seed,days,start){
  const rnd=mulberry32(seed);let p=start||100;let vol=0.012;let drift=0.0004;
  const out=[p];
  for(let i=1;i<days;i++){
    if(rnd()<0.02){drift=(rnd()-0.45)*0.004;vol=0.006+rnd()*0.02}
    vol=Math.min(0.035,Math.max(0.005,vol+(rnd()-0.5)*0.002));
    p=p*Math.exp(drift-0.5*vol*vol+vol*gauss(rnd));
    out.push(Math.round(p*100)/100);
  }
  return out;
}
// windowed supervised framing on log returns
function frame(series,window){
  const rets=[];for(let i=1;i<series.length;i++)rets.push(Math.log(series[i]/series[i-1]));
  const mu=rets.reduce((s,x)=>s+x,0)/rets.length;
  const sd=Math.sqrt(rets.reduce((s,x)=>s+(x-mu)*(x-mu),0)/rets.length)||1;
  const X=[],Y=[];
  for(let i=window;i<rets.length;i++){X.push(rets.slice(i-window,i).map(r=>(r-mu)/(3*sd)));Y.push((rets[i]-mu)/(3*sd))}
  return {X,Y,mu,sd,rets};
}
async function trainLSTM(fr,epochs,onEpoch){
  await tf.setBackend('cpu');await tf.ready();
  const w=fr.X[0].length;
  const m=tf.sequential();
  m.add(tf.layers.lstm({units:16,inputShape:[w,1],returnSequences:false}));
  m.add(tf.layers.dense({units:1}));
  m.compile({optimizer:tf.train.adam(0.02),loss:'meanSquaredError'});
  const xs=tf.tensor3d(fr.X.map(r=>r.map(v=>[v])));
  const ys=tf.tensor2d(fr.Y.map(v=>[v]));
  await m.fit(xs,ys,{epochs,batchSize:32,shuffle:true,callbacks:{onEpochEnd:(e,l)=>onEpoch&&onEpoch(e,l)}});
  xs.dispose();ys.dispose();
  return m;
}
function forecast(model,fr,series,window,horizon){
  const hist=[...fr.rets];
  const lastPrice=series[series.length-1];
  const out=[];let price=lastPrice;
  const w=window;
  for(let h=0;h<horizon;h++){
    const win=hist.slice(-w).map(r=>(r-fr.mu)/(3*fr.sd));
    const t=tf.tensor3d([win.map(v=>[v])]);
    const pred=model.predict(t);
    const z=pred.dataSync()[0]*3*fr.sd+fr.mu;
    t.dispose();pred.dispose();
    price=price*Math.exp(z);
    out.push(Math.round(price*100)/100);
    hist.push(z);
  }
  return out;
}
function naiveForecast(series,horizon){
  // random walk with drift = mean historical log return
  const rets=[];for(let i=1;i<series.length;i++)rets.push(Math.log(series[i]/series[i-1]));
  const mu=rets.reduce((s,x)=>s+x,0)/rets.length;
  let p=series[series.length-1];const out=[];
  for(let h=0;h<horizon;h++){p=p*Math.exp(mu);out.push(Math.round(p*100)/100)}
  return out;
}
function metrics(actual,pred){
  const n=Math.min(actual.length,pred.length);let mae=0,mse=0;
  for(let i=0;i<n;i++){const e=actual[i]-pred[i];mae+=Math.abs(e);mse+=e*e}
  return {mae:mae/n,rmse:Math.sqrt(mse/n)};
}
window.StockEngine={synthStock,frame,trainLSTM,forecast,naiveForecast,metrics};
})();
