/* stock-lstm-forecaster UI */
(function(){
'use strict';
const $=id=>document.getElementById(id);
let model=null,state=null,busy=false;
const DEMOS={
  steady:{seed:7,days:300,start:100,name:'"ACME" — steady climber'},
  volatile:{seed:21,days:300,start:80,name:'"VOLTA" — wild rides'},
  falling:{seed:33,days:300,start:150,name:'"DRIFT" — slow bleed'}
};
function fmt(x,d){return x==null?'—':x.toLocaleString(undefined,{maximumFractionDigits:d==null?2:d})}
function parseSeries(t){const v=t.split(/[\s,;\n\t]+/).map(x=>parseFloat(x)).filter(x=>Number.isFinite(x)&&x>0);return v.length>=60?v:null}
async function run(){
  if(busy)return;busy=true;
  $('run').disabled=true;
  try{
    let series,label;
    const pasted=parseSeries($('paste').value.trim());
    if(pasted){series=pasted;label='your pasted series'}
    else{const d=DEMOS[$('demo').value];series=StockEngine.synthStock(d.seed,d.days,d.start);label=d.name}
    const window=20,horizon=10;
    const holdN=30;
    const train=series.slice(0,series.length-holdN),hold=series.slice(series.length-holdN);
    $('prog-wrap').classList.remove('hidden');
    const prog=(p,msg)=>{$('prog-bar').style.width=Math.round(p)+'%';$('prog-msg').textContent=msg};
    // BACKTEST: train on train-part, predict hold
    const frBT=StockEngine.frame(train,window);
    prog(5,'Training LSTM for backtest (CPU)…');
    const mBT=await StockEngine.trainLSTM(frBT,12,(e,l)=>prog(5+(e+1)/12*45,`Backtest epoch ${e+1}/12 — loss ${l.loss.toFixed(5)}`));
    const lstmBT=StockEngine.forecast(mBT,frBT,train,window,holdN);
    const naiveBT=StockEngine.naiveForecast(train,holdN);
    const mL=StockEngine.metrics(hold,lstmBT),mN=StockEngine.metrics(hold,naiveBT);
    mBT.dispose();
    // FINAL: train on full series, forecast future
    const frFull=StockEngine.frame(series,window);
    prog(55,'Training final model on all data…');
    model=await StockEngine.trainLSTM(frFull,12,(e,l)=>prog(55+(e+1)/12*40,`Final epoch ${e+1}/12 — loss ${l.loss.toFixed(5)}`));
    const future=StockEngine.forecast(model,frFull,series,window,horizon);
    prog(100,'Done.');
    state={series,label,hold,holdN,lstmBT,naiveBT,mL,mN,future,horizon};
    $('results').classList.remove('hidden');
    render();
    $('results').scrollIntoView({behavior:'smooth'});
  }catch(e){console.error(e);pkToast('Training failed: '+e.message)}
  finally{busy=false;$('run').disabled=false}
}
function render(){
  const s=state;
  const winner=s.mL.rmse<=s.mN.rmse?'LSTM':'Naive';
  $('bt-note').innerHTML=`<span class="pill ${winner==='LSTM'?'ok':'warn'}">${winner==='LSTM'?'LSTM won this backtest':'Naive baseline won — the LSTM found no real edge'}</span> <span class="hint">backtested on the last ${s.holdN} points of ${s.label}</span>`;
  $('metric-tbl').innerHTML=
    `<tr class="${winner==='LSTM'?'row-win':''}"><td><strong>LSTM (this app)</strong></td><td class="mono">${fmt(s.mL.mae)}</td><td class="mono">${fmt(s.mL.rmse)}</td></tr>`+
    `<tr class="${winner==='Naive'?'row-win':''}"><td><strong>Naive (drift) baseline</strong></td><td class="mono">${fmt(s.mN.mae)}</td><td class="mono">${fmt(s.mN.rmse)}</td></tr>`;
  $('fc-tbl').innerHTML=s.future.map((f,i)=>`<tr><td class="mono">+${i+1}</td><td class="mono"><strong>${fmt(f)}</strong></td><td class="mono hint">${fmt((f/s.series[s.series.length-1]-1)*100,1)}%</td></tr>`).join('');
  drawChart();
}
function drawChart(){
  const s=state;const cv=$('chart');const ctx=cv.getContext('2d');
  const W=cv.width=cv.clientWidth*2,H=cv.height=420;
  const dark=document.documentElement.dataset.theme==='dark';
  ctx.clearRect(0,0,W,H);
  const hist=s.series.slice(-90);
  const all=[...hist,...s.lstmBT,...s.naiveBT,...s.future];
  const lo=Math.min(...all),hi=Math.max(...all),rng=hi-lo||1;
  const histN=hist.length,total=histN+s.holdN+s.future.length;
  const X=i=>50+i/(total-1)*(W-130),Y=v=>H-36-((v-lo)/rng)*(H-70);
  ctx.strokeStyle=dark?'#1e2c49':'#e9eef6';ctx.fillStyle=dark?'#8ea2c4':'#5d6f8e';ctx.font='19px JetBrains Mono, monospace';ctx.lineWidth=1;
  for(let g=0;g<=4;g++){const v=lo+rng*g/4;ctx.beginPath();ctx.moveTo(50,Y(v));ctx.lineTo(W-70,Y(v));ctx.stroke();ctx.fillText(fmt(v),W-64,Y(v)+6)}
  const line=(arr,off,color,width,dash)=>{ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash||[]);ctx.beginPath();arr.forEach((v,i)=>{const x=X(off+i),y=Y(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.setLineDash([])};
  line(hist,0,dark?'#c2cfe8':'#0c2340',2.4);
  const accent=getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()||'#4f46e5';
  // backtest predictions overlay on the holdout zone
  line(s.lstmBT,histN-s.holdN,accent,2.6,[2,0]);
  line(s.naiveBT,histN-s.holdN,'#b45309',2,[6,5]);
  // future forecast continues from series end
  line(s.future,histN-1+0,accent,3,[9,6]);
  ctx.strokeStyle=dark?'#2a3c60':'#c9d5e8';ctx.lineWidth=1.6;
  ctx.beginPath();ctx.moveTo(X(histN-s.holdN),10);ctx.lineTo(X(histN-s.holdN),H-30);ctx.stroke();
  ctx.beginPath();ctx.moveTo(X(histN-1),10);ctx.lineTo(X(histN-1),H-30);ctx.stroke();
  ctx.fillStyle=dark?'#8ea2c4':'#5d6f8e';
  ctx.fillText('history',X(10)-10,26);ctx.fillText('backtest',X(histN-s.holdN+2),26);ctx.fillText('future',X(histN+2),26);
  // legend
  const lg=[['Actual',dark?'#c2cfe8':'#0c2340'],['LSTM',accent],['Naive','#b45309']];
  lg.forEach((l,i)=>{ctx.fillStyle=l[1];ctx.fillRect(50+i*150,H-22,18,5);ctx.fillStyle=dark?'#8ea2c4':'#5d6f8e';ctx.fillText(l[0],74+i*150,H-14)});
}
document.addEventListener('DOMContentLoaded',()=>{
  $('run').onclick=run;
  $('demo').onchange=()=>{$('paste').value=''};
  new MutationObserver(()=>{if(state)drawChart()}).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  window.addEventListener('resize',()=>{if(state)drawChart()});
});
})();
