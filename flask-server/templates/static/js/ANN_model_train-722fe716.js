import{e as q,p as i,f as z,k as p,o as _,c as J,j as l,A as t,h,l as s,u as Q,K as T,b as W,ay as r}from"./index-7d70ad9a.js";const oe=q({__name:"ANN_model_train",setup(Z){const m=i(0),d=i({cycles:"",epoch:"",feature_add_num:""}),v=i([]),k=i([]),C=i(!1),V=i(!1),U=i([]),g=i([]),E=i(!1),D=i(!1),L=o=>e=>{const n=e.target.value;d.value[o]=n.replace(/\D/g,"")},A=async()=>{if(U.value.length===0||g.value.length===0){r.error("请上传文件");return}E.value=!0;const o=new FormData;o.append("model",U.value[0]),o.append("data_pre",g.value[0]);try{const e=await fetch("/predict",{method:"POST",body:o});return D.value=!0,e}catch{r.error("发生错误")}finally{E.value=!1}},N=()=>{const o=parseInt(d.value.feature_add_num);if(isNaN(o)||o<0||o>10){r.error("Feature Add Num must be a number between 0 and 10");return}if(v.value.length===0||k.value.length===0){r.error("请上传文件");return}G()},R=()=>{m.value++},M=(o,e,n)=>{U.value.push(e.raw),r.success("pth file uploaded successfully")},O=(o,e,n)=>{g.value.push(e.raw),r.success("model file uploaded successfully")},H=(o,e,n)=>{v.value.push(e.raw),r.success("Data file uploaded successfully")},S=(o,e,n)=>{k.value.push(e.raw),r.success("Data2 file uploaded successfully")},G=async()=>{if(!F)return;C.value=!0;const o=new FormData;o.append("cycles",d.value.cycles),o.append("epoch",d.value.epoch),o.append("feature_add_num",d.value.feature_add_num),o.append("data",v.value[0]),o.append("data2",k.value[0]);try{(await await fetch("/train",{method:"POST",body:o})).status==200?(r.success("Model trained successfully"),V.value=!0):r.error("训练失败，请检查文件是否有误")}catch(e){console.error("Error:",e),r.error("训练失败，请检查文件是否有误")}finally{C.value=!1}},b=(o,e)=>{if(o.name.split(".")[1]!==e)return r.error(`请上传.${e}文件`),!1},B=async()=>{try{const e=await(await fetch("/download_results",{method:"GET"})).blob(),n=window.URL.createObjectURL(e),a=document.createElement("a");a.href=n,a.setAttribute("download","results.csv"),document.body.appendChild(a),a.click(),r.success("Results CSV downloaded successfully")}catch(o){console.error("Error:",o),r.error("Failed to download results CSV")}},X=async()=>{try{const e=await(await fetch("/download_model",{method:"GET"})).blob(),n=window.URL.createObjectURL(e),a=document.createElement("a");a.href=n,a.setAttribute("download","best_model.pth"),document.body.appendChild(a),a.click(),r.success("Model PTH downloaded successfully")}catch(o){console.error("Error:",o),r.error("Failed to download model PTH")}},Y=async()=>{try{const e=await(await fetch("/download_joblibX",{method:"GET"})).blob(),n=window.URL.createObjectURL(e),a=document.createElement("a");a.href=n,a.setAttribute("download","X_scaler.joblib"),document.body.appendChild(a),a.click(),r.success("Model PTH downloaded successfully")}catch(o){console.error("Error:",o),r.error("Failed to download model PTH")}},I=async()=>{try{const e=await(await fetch("/download_joblibY",{method:"GET"})).blob(),n=window.URL.createObjectURL(e),a=document.createElement("a");a.href=n,a.setAttribute("download","Y_scaler.joblib"),document.body.appendChild(a),a.click(),r.success("Model PTH downloaded successfully")}catch(o){console.error("Error:",o),r.error("Failed to download model PTH")}},$=async()=>{try{const e=await(await fetch("/download_predict_file",{method:"GET"})).blob(),n=window.URL.createObjectURL(e),a=document.createElement("a");a.href=n,a.setAttribute("download","predicted_output_file.xlsx"),document.body.appendChild(a),a.click(),r.success("predicted file downloaded successfully")}catch(o){console.error("Error:",o),r.error("Failed to download model file")}},F=z(()=>d.value.cycles&&d.value.epoch&&d.value.feature_add_num),y=(o,e,n)=>{r.error("上传文件发生错误，请检查文件是否正确")};return(o,e)=>{const n=p("el-step"),a=p("el-steps"),x=p("el-input"),f=p("el-form-item"),u=p("el-button"),w=p("el-upload"),K=p("el-form"),j=p("el-main"),P=p("el-container");return _(),J("div",null,[l(a,{active:m.value,"finish-status":"success","align-center":"",style:{"margin-bottom":"20px"}},{default:t(()=>[l(n,{title:"第一步"}),l(n,{title:"第二步"})]),_:1},8,["active"]),m.value===0?(_(),h(P,{key:0},{default:t(()=>[l(j,null,{default:t(()=>[l(K,{ref:"form",model:d.value,"label-width":"120px"},{default:t(()=>[l(f,{label:"Cycles"},{default:t(()=>[l(x,{onInput:e[0]||(e[0]=c=>L("cycles")),modelValue:d.value.cycles,"onUpdate:modelValue":e[1]||(e[1]=c=>d.value.cycles=c),modelModifiers:{trim:!0},placeholder:"请输入整数",min:"0",type:"number",step:"1"},null,8,["modelValue"])]),_:1}),l(f,{label:"Epoch"},{default:t(()=>[l(x,{modelValue:d.value.epoch,"onUpdate:modelValue":e[2]||(e[2]=c=>d.value.epoch=c),modelModifiers:{trim:!0},placeholder:"请输入整数",min:"0",type:"number"},null,8,["modelValue"])]),_:1}),l(f,{label:"FeatureAddNum"},{default:t(()=>[l(x,{modelValue:d.value.feature_add_num,"onUpdate:modelValue":e[3]||(e[3]=c=>d.value.feature_add_num=c),modelModifiers:{trim:!0},placeholder:"请输入0-10的数",type:"number",min:"0",max:"10"},null,8,["modelValue"])]),_:1}),l(f,{label:"Data File(.xlsx)"},{default:t(()=>[l(w,{action:"/upload","on-success":H,"before-upload":c=>b(c,"xlsx"),limit:1,"on-error":y},{default:t(()=>[l(u,{slot:" trigger",type:"primary"},{default:t(()=>[s("Upload")]),_:1}),s("(最多上传一个文件) ")]),_:1},8,["before-upload"])]),_:1}),l(f,{label:"Data2 File(.csv)"},{default:t(()=>[l(w,{"on-success":S,action:"/upload","before-upload":c=>b(c,"csv"),limit:1,"on-error":y},{default:t(()=>[l(u,{slot:"trigger",type:"primary"},{default:t(()=>[s("Upload")]),_:1}),s("(最多上传一个文件) ")]),_:1},8,["before-upload"])]),_:1}),l(f,null,{default:t(()=>[l(u,{type:"primary",onClick:N,loading:C.value,disabled:!Q(F)},{default:t(()=>[s("Train Model")]),_:1},8,["loading","disabled"])]),_:1}),V.value?(_(),h(f,{key:0},{default:t(()=>[l(u,{type:"success",onClick:B},{default:t(()=>[s("Download CSV")]),_:1}),l(u,{type:"success",onClick:X},{default:t(()=>[s("Download PTH")]),_:1}),l(u,{type:"success",onClick:Y},{default:t(()=>[s("Download joblibX")]),_:1}),l(u,{type:"success",onClick:I},{default:t(()=>[s("Download joblibY")]),_:1}),l(u,{type:"primary",onClick:R},{default:t(()=>[s("下一步")]),_:1})]),_:1})):T("",!0)]),_:1},8,["model"])]),_:1})]),_:1})):m.value===1?(_(),h(P,{key:1},{default:t(()=>[l(j,null,{default:t(()=>[W("div",null,[l(w,{action:"/upload","on-success":M,"auto-upload":!0,"before-upload":c=>b(c,"pth"),limit:1,"on-error":y},{default:t(()=>[s(" 模型文件(.pth)："),l(u,{type:"primary"},{default:t(()=>[s("上传模型文件")]),_:1}),s("(最多上传一个文件) ")]),_:1},8,["before-upload"]),l(w,{action:"/upload","on-success":O,limit:1,"auto-upload":!0,"before-upload":c=>b(c,"xlsx"),"on-error":y},{default:t(()=>[s(" 数据文件(.xlsx)："),l(u,{type:"primary"},{default:t(()=>[s("上传数据文件")]),_:1}),s("(最多上传一个文件) ")]),_:1},8,["before-upload"]),l(u,{onClick:A,loading:E.value,type:"primary"},{default:t(()=>[s("预测")]),_:1},8,["loading"]),D.value?(_(),h(u,{key:0,type:"success",onClick:$},{default:t(()=>[s("下载预测结果")]),_:1})):T("",!0),l(u,{type:"primary",onClick:e[4]||(e[4]=c=>m.value--)},{default:t(()=>[s("返回上一步")]),_:1})])]),_:1})]),_:1})):T("",!0)])}}});export{oe as default};