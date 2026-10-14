// Optional developer check: npm install @napi-rs/canvas, then
// node audit/check_view.cjs examples/checked_result/visualization.html OUTPUT_DIR
// Runs the actual viewer script with native canvases; not a browser layout test.
const fs=require('fs'),path=require('path'),vm=require('vm');
const mod=require.resolve('@napi-rs/canvas',{paths:[process.cwd(),...(process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES?[process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES]:[])]});
const {createCanvas}=require(mod);
const html=fs.readFileSync(process.argv[2],'utf8'),out=process.argv[3];
fs.mkdirSync(out,{recursive:true});
const json=html.match(/<script id="tensor-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const elements={'tensor-data':{textContent:json}};
for(const id of ['status','height','slice-label','point-note'])elements[id]={value:'0',textContent:''};
const ids=['raw','remaining','vector','matrix','slice'];
for(const id of ids){const c=createCanvas(450,300);c.getBoundingClientRect=()=>({width:450});c.setPointerCapture=()=>{};elements[id]=c;}
const events={};vm.runInNewContext(script,{document:{getElementById:id=>{if(!(id in elements))throw Error('Missing element '+id);return elements[id];}},addEventListener:(type,fn)=>events[type]=fn,console},{timeout:5000});
const stats={};for(const id of ids){const c=elements[id];const pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;const colors=new Set();for(let i=0;i<pixels.length;i+=4)colors.add([pixels[i],pixels[i+1],pixels[i+2],pixels[i+3]].join(','));if(colors.size<8)throw Error(id+' rendered too few colors');stats[id]={width:c.width,height:c.height,color_count:colors.size};fs.writeFileSync(path.join(out,id+'.png'),c.toBuffer('image/png'));}
const rawBefore=elements.raw.toBuffer('image/png');elements.raw.onpointerdown({clientX:20,clientY:20,pointerId:1});elements.raw.onpointermove({clientX:80,clientY:50});elements.raw.onpointerup();
const rotationChanged=!rawBefore.equals(elements.raw.toBuffer('image/png'));if(!rotationChanged)throw Error('Rotation did not change the plot');
const before=elements.slice.toBuffer('image/png');elements.height.value='7';elements.height.oninput();const sliceChanged=!before.equals(elements.slice.toBuffer('image/png'));if(!sliceChanged)throw Error('Slider did not change slice');
if(!elements['slice-label'].textContent.startsWith('k = 7'))throw Error('Slice label wrong');
const report={status:'passed',engine:'actual embedded JavaScript + native canvas, simulated DOM',plots:stats,rotationChanged,sliceChanged,scriptErrors:0,fullBrowserLayoutTest:false};
fs.writeFileSync(path.join(out,'VIEW_CHECK.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
