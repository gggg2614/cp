import{e as g,p as l,o as p,c as k,j as _,A as v,l as f,u as d,av as E,t as B,aw as C,h as D,ax as L,K as N,b as i,ay as u}from"./index-94bb87a7.js";const U=i("i",{class:"el-icon-upload"},null,-1),V=i("div",{class:"el-upload__text"},[f("将文件拖到此处，或"),i("em",null,"点击上传")],-1),S=i("span",{style:{"margin-top":"10px","font-size":"12px",color:"#999"}}," 高交叉率有助于增加种群的多样性 ",-1),A=g({__name:"feature_set_choose",setup(T){const o=l([]),r=l(!1),n=l("开始特征选择"),m=l(!1),s=l(!0),b=(a,e,t)=>{o.value.push(e.raw),u({type:"success",message:"上传成功"}),s.value=!1},h=a=>{if(a.name.split(".")[1]!=="xlsx")return u.error("请上传.xlsx文件"),!1},x=async()=>{if(r.value=!0,n.value="处理中...",s.value=!0,!o.value||o.value.length===0){u.error("请上传文件。");return}const a=new FormData;a.append("file",o.value[0]);try{const e=await fetch("http://localhost:5000/feature_selection",{method:"POST",body:a});if(!e.ok)throw new Error("网络响应不正常");const t=await e.blob();y(t)}catch(e){console.error("发生了与 fetch 操作相关的问题：",e),r.value=!1,n.value="开始特征选择",s.value=!1}},w=(a,e,t)=>{u.error("上传文件发生错误，请检查文件是否正确")},y=a=>{const e=new Blob([a],{type:"text/csv"}),t=window.URL.createObjectURL(e),c=document.createElement("a");c.href=t,c.setAttribute("download","results.csv"),document.body.appendChild(c),c.click(),r.value=!1,n.value="开始特征选择",s.value=!1};return(a,e)=>(p(),k("div",null,[_(d(E),{class:"upload-demo","on-error":w,drag:"",action:"/upload","show-file-list":!1,"on-success":b,"before-upload":h,disabled:m.value,limit:1},{default:v(()=>[U,V,f(" (最多上传一个文件) ")]),_:1},8,["disabled"]),_(d(C),{type:"primary",onClick:x,disabled:s.value},{default:v(()=>[f(B(n.value),1)]),_:1},8,["disabled"]),S,r.value?(p(),D(d(L),{key:0,text:"处理中...",fullscreen:""})):N("",!0)]))}});export{A as default};
