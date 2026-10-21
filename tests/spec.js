module.exports=async(page)=>{
  await page.click('#run');
  await page.waitForSelector('#results:not(.hidden)',{timeout:300000});
  const note=await page.$eval('#bt-note',e=>e.textContent);
  const rows=await page.$$eval('#fc-tbl tr',e=>e.length);
  if(rows!==10)throw new Error('expected 10 forecast rows, got '+rows);
  const metrics=await page.$$eval('#metric-tbl tr',e=>e.length);
  if(metrics!==2)throw new Error('expected 2 model rows');
  const winner=await page.$eval('.row-win td',e=>e.textContent);
  const painted=await page.evaluate(()=>{const c=document.getElementById('chart');const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=3;i<d.length;i+=4){if(d[i]>0)n++}return n});
  if(painted<1000)throw new Error('chart empty');
  return `backtest done, winner=${winner.trim()}, 10 forecasts, chart painted, note="${note.slice(0,60)}"`;
};
