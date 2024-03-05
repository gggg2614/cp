const Q=Vue.createElementVNode("div",{style:{"margin-top":"10px","font-size":"12px",color:"#999"}}," 对于小数据集来说，相对较小的种群就可以覆盖解空间 ",-1),W=Vue.createElementVNode("div",{style:{"margin-top":"10px","font-size":"12px",color:"#999"}}," 设置一个稍高的迭代次数可以深化算法的探索与优化 ",-1),Y=Vue.createElementVNode("div",{style:{"margin-top":"10px","font-size":"12px",color:"#999"}}," 简约系数用于避免过于复杂的解。考虑到数据规模较小，可以设置一个中等的值，如0.01到01，以鼓励算法寻找更简洁的解。 ",-1),Z=Vue.createElementVNode("div",{style:{"margin-top":"10px","font-size":"12px",color:"#999"}}," 由于数据集规模较小，可以考虑使用较大的样本比例，甚至是100%，以充分利用所有可用数据。 ",-1),ee=Vue.createElementVNode("div",{style:{"margin-top":"10px","font-size":"12px",color:"#999"}}," 高交叉率有助于增加种群的多样性 ",-1),te=Vue.createElementVNode("div",{style:{"margin-top":"10px","font-size":"12px",color:"#999"}}," 变异率可以设置得相对较高，以促进解空间的探索 ",-1),oe=Vue.createElementVNode("div",{style:{"margin-top":"10px","font-size":"12px",color:"#999"}}," 由于提升变异可以帮助简化解的结构，推荐的概率设置为0.05到0.1之间。 ",-1),le=Vue.createElementVNode("div",{style:{"margin-top":"10px","font-size":"12px",color:"#999"}}," 点变异直接修改程序树的细节，建议的概率也是0.05到0.1之间。 ",-1),ue=Vue.createElementVNode("div",{class:"el-upload__tip",slot:"tip"},"支持上传 XLSX/XLS 文件",-1),ae=Vue.createElementVNode("h3",null,"返回的结果：",-1),re=Vue.createElementVNode("h3",null,"生成的特征：",-1),ne={key:0},de={key:1},Ve=Vue.createElementVNode("p",null,"暂无特征数据。",-1),se=[Ve],ce=Vue.defineComponent({__name:"symbolic_regression",setup(ie){const d=Vue.ref(0),m=Vue.ref("1"),c=Vue.ref([]),V=Vue.ref({}),f=Vue.ref(!1),v=Vue.ref(!1),O="http://localhost:5000/upload",x=Vue.ref(20),h=Vue.ref(0),_=Vue.ref(1),N=Vue.ref(1),w=Vue.ref("1"),C=Vue.ref(.3),y=Vue.ref(.2),b=Vue.ref(.1),g=Vue.ref(.15),M=Vue.ref(.05),k=Vue.ref(.1),E=Vue.ref(.05),S=Vue.ref(.1),B=Vue.ref(1e-6),U=Vue.ref(.099999),P=Vue.ref(["add","sub","mul","div","sqrt","log","abs","neg","inv"]),$=Vue.ref(""),j=t=>{const e=t.best_params;V.value={};for(const l in e)V.value[l]=e[l];V.value.best_score=t.best_score},R=async()=>{try{const e=await(await fetch("http://localhost:5000/download_txt",{method:"GET"})).blob(),l=window.URL.createObjectURL(e),n=document.createElement("a");n.href=l,n.setAttribute("download","symbolic_transformer_formulas.txt"),document.body.appendChild(n),n.click(),ElementPlus.ElMessage.success("txt downloaded successfully")}catch(t){console.error("Error:",t),ElementPlus.ElMessage.error("Failed to download model txt")}},F=async()=>{try{const e=await(await fetch("http://localhost:5000/download_txt_names",{method:"GET"})).blob(),l=window.URL.createObjectURL(e),n=document.createElement("a");n.href=l,n.setAttribute("download","symbolic_transformer_formulas_with_feature_names.txt"),document.body.appendChild(n),n.click(),ElementPlus.ElMessage.success("txtNames downloaded successfully")}catch(t){console.error("Error:",t),ElementPlus.ElMessage.error("Failed to download model txtNames")}},D=async()=>{const t=new FormData;t.append("file",c.value[0]),t.append("best_params",JSON.stringify(V.value)),fetch("http://localhost:5000/get_gp_features",{method:"POST",body:t}).then(e=>{if(!e.ok)throw new Error("网络响应不正常");return e.json()})},H=()=>{if(parseInt(m.value)<=0){ElementPlus.ElMessage.error("请输入大于0的整数");return}(async()=>{if(await Vue.nextTick(),m.value.trim()===""){ElementPlus.ElMessage.error("请输入搜索次数。");return}if(!/^\d+$/.test(m.value)){ElementPlus.ElMessage.error("搜索次数必须为整数。");return}d.value++})()},z=()=>{c.value[0]==null,d.value--},A=t=>{if(t.name.split(".")[1]!=="xlsx")return ElementPlus.ElMessage.error("请上传.xlsx文件"),!1},X=(t,e,l)=>{c.value.push(e.raw),ElementPlus.ElMessage({type:"success",message:"上传成功"})},G=()=>{fetch("http://localhost:5000/generate_features",{method:"POST",body:JSON.stringify({best_params:V.value}),headers:{"Content-Type":"application/json"}}).then(t=>{if(!t.ok)throw new Error("网络响应不正常");return t.blob()}).then(t=>{const e=window.URL.createObjectURL(new Blob([t])),l=document.createElement("a");v.value=!0,l.href=e,l.setAttribute("download","features.csv"),document.body.appendChild(l),l.click(),l.parentNode.removeChild(l)}).catch(t=>{console.error("发生了与 fetch 操作相关的问题：",t)}).finally(()=>{v.value=!1})},J=()=>{if(!c.value||c.value.length===0){ElementPlus.ElMessage.error("请上传文件。");return}const t=new FormData;t.append("file",c.value[0]),t.append("n_searches",m.value),t.append("population_size",`${x.value},${h.value}`),t.append("generations",`${_.value},${N.value}`),t.append("maxSamples",w.value),t.append("pCrossover",`${C.value},${y.value}`),t.append("pSubtreeMutation",`${b.value},${g.value}`),t.append("pHoistMutation",`${M.value},${k.value}`),t.append("pPointMutation",`${E.value},${S.value}`),t.append("parsimonyCoefficient",`${B.value},${U.value}`),t.append("functionSet",P.value),f.value=!0,fetch("http://localhost:5000/file1",{method:"POST",body:t}).then(e=>{if(!e.ok)throw new Error("网络响应不正常");return e.json()}).then(e=>{j(e),D(),d.value=2}).catch(e=>{console.error("发生了问题：",e)}).finally(()=>{f.value=!1})},q=(t,e,l)=>{ElementPlus.ElMessage.error("上传文件发生错误，请检查文件是否正确")};return(t,e)=>{const l=Vue.resolveComponent("el-step"),n=Vue.resolveComponent("el-steps"),u=Vue.resolveComponent("el-input"),r=Vue.resolveComponent("el-form-item"),a=Vue.resolveComponent("el-col"),s=Vue.resolveComponent("el-row"),i=Vue.resolveComponent("el-button"),L=Vue.resolveComponent("el-form"),T=Vue.resolveComponent("el-card"),I=Vue.resolveComponent("el-upload");return Vue.openBlock(),Vue.createElementBlock("div",null,[Vue.createVNode(n,{active:d.value,"finish-status":"success","align-center":"",style:{"margin-bottom":"20px"}},{default:Vue.withCtx(()=>[Vue.createVNode(l,{title:"第一步"}),Vue.createVNode(l,{title:"第二步"}),Vue.createVNode(l,{title:"下载结果"})]),_:1},8,["active"]),d.value===0?(Vue.openBlock(),Vue.createBlock(T,{key:0},{default:Vue.withCtx(()=>[Vue.createVNode(L,{"label-width":"120px",model:V.value},{default:Vue.withCtx(()=>[Vue.createVNode(r,{label:"输入搜索次数",prop:"nSearches"},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:m.value,"onUpdate:modelValue":e[0]||(e[0]=o=>m.value=o),modelModifiers:{trim:!0},placeholder:"输入搜索次数",type:"number",min:"0"},null,8,["modelValue"])]),_:1}),Vue.createVNode(r,{label:"符号集",prop:"nSearches"},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:P.value,"onUpdate:modelValue":e[1]||(e[1]=o=>P.value=o),modelModifiers:{trim:!0},placeholder:"输入搜索次数"},null,8,["modelValue"])]),_:1}),Vue.createVNode(r,{label:"种群大小",prop:"populationSize"},{default:Vue.withCtx(()=>[Vue.createVNode(s,{gutter:20},{default:Vue.withCtx(()=>[Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:x.value,"onUpdate:modelValue":e[2]||(e[2]=o=>x.value=o),modelModifiers:{trim:!0},placeholder:"最小值",type:"number",min:"40"},null,8,["modelValue"])]),_:1}),Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:h.value,"onUpdate:modelValue":e[3]||(e[3]=o=>h.value=o),modelModifiers:{trim:!0},placeholder:"宽度",type:"number",min:"40"},null,8,["modelValue"])]),_:1})]),_:1}),Q]),_:1}),Vue.createVNode(r,{label:"迭代代数",prop:"generations"},{default:Vue.withCtx(()=>[Vue.createVNode(s,{gutter:20},{default:Vue.withCtx(()=>[Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:_.value,"onUpdate:modelValue":e[4]||(e[4]=o=>_.value=o),modelModifiers:{trim:!0},placeholder:"最小值",type:"number",min:"50"},null,8,["modelValue"])]),_:1}),Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:N.value,"onUpdate:modelValue":e[5]||(e[5]=o=>N.value=o),modelModifiers:{trim:!0},placeholder:"宽度",type:"number",min:"50"},null,8,["modelValue"])]),_:1})]),_:1}),W]),_:1}),Vue.createVNode(r,{label:"简约系数",prop:"parsimonyCoefficient"},{default:Vue.withCtx(()=>[Vue.createVNode(s,{gutter:20},{default:Vue.withCtx(()=>[Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:B.value,"onUpdate:modelValue":e[6]||(e[6]=o=>B.value=o),modelModifiers:{trim:!0},placeholder:"最小值",type:"number",min:"0.000001"},null,8,["modelValue"])]),_:1}),Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:U.value,"onUpdate:modelValue":e[7]||(e[7]=o=>U.value=o),modelModifiers:{trim:!0},placeholder:"宽度",type:"number"},null,8,["modelValue"])]),_:1})]),_:1}),Y]),_:1}),Vue.createVNode(r,{label:"最大样本数",prop:"maxSamples"},{default:Vue.withCtx(()=>[Vue.createVNode(a,{span:5.5},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:w.value,"onUpdate:modelValue":e[8]||(e[8]=o=>w.value=o),modelModifiers:{trim:!0},placeholder:"最大样本数",type:"number",min:"1"},null,8,["modelValue"])]),_:1},8,["span"]),Z]),_:1}),Vue.createVNode(r,{label:"交叉概率",prop:"pCrossover"},{default:Vue.withCtx(()=>[Vue.createVNode(s,{gutter:20},{default:Vue.withCtx(()=>[Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:C.value,"onUpdate:modelValue":e[9]||(e[9]=o=>C.value=o),modelModifiers:{trim:!0},placeholder:"最小值",type:"number",min:"0.3",max:"0.5"},null,8,["modelValue"])]),_:1}),Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:y.value,"onUpdate:modelValue":e[10]||(e[10]=o=>y.value=o),modelModifiers:{trim:!0},placeholder:"宽度",type:"number",min:"0.3",max:"0.5"},null,8,["modelValue"])]),_:1})]),_:1}),ee]),_:1}),Vue.createVNode(r,{label:"子树变异概率",prop:"pSubtreeMutation"},{default:Vue.withCtx(()=>[Vue.createVNode(s,{gutter:20},{default:Vue.withCtx(()=>[Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:b.value,"onUpdate:modelValue":e[11]||(e[11]=o=>b.value=o),modelModifiers:{trim:!0},placeholder:"最小值",type:"number",min:"0.1",max:"0.25"},null,8,["modelValue"])]),_:1}),Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:g.value,"onUpdate:modelValue":e[12]||(e[12]=o=>g.value=o),modelModifiers:{trim:!0},placeholder:"宽度",type:"number",min:"0.1",max:"0.25"},null,8,["modelValue"])]),_:1})]),_:1}),te]),_:1}),Vue.createVNode(r,{label:"提升变异",prop:"pHoistMutation"},{default:Vue.withCtx(()=>[Vue.createVNode(s,{gutter:20},{default:Vue.withCtx(()=>[Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:M.value,"onUpdate:modelValue":e[13]||(e[13]=o=>M.value=o),modelModifiers:{trim:!0},placeholder:"最小值",type:"number",min:"0.05",max:"0.15"},null,8,["modelValue"])]),_:1}),Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:k.value,"onUpdate:modelValue":e[14]||(e[14]=o=>k.value=o),modelModifiers:{trim:!0},placeholder:"宽度",type:"number",min:"0.05",max:"0.15"},null,8,["modelValue"])]),_:1})]),_:1}),oe]),_:1}),Vue.createVNode(r,{label:"点变异",prop:"pPointMutation"},{default:Vue.withCtx(()=>[Vue.createVNode(s,{gutter:20},{default:Vue.withCtx(()=>[Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:E.value,"onUpdate:modelValue":e[15]||(e[15]=o=>E.value=o),modelModifiers:{trim:!0},placeholder:"最小值",type:"number",min:"0.05",max:"0.15"},null,8,["modelValue"])]),_:1}),Vue.createVNode(a,{span:12},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:S.value,"onUpdate:modelValue":e[16]||(e[16]=o=>S.value=o),modelModifiers:{trim:!0},placeholder:"宽度",type:"number",min:"0.05",max:"0.15"},null,8,["modelValue"])]),_:1})]),_:1}),le]),_:1}),Vue.createVNode(r,null,{default:Vue.withCtx(()=>[Vue.createVNode(i,{type:"primary",onClick:H},{default:Vue.withCtx(()=>[Vue.createTextVNode("下一步")]),_:1})]),_:1})]),_:1},8,["model"])]),_:1})):Vue.createCommentVNode("",!0),d.value===1?(Vue.openBlock(),Vue.createBlock(I,{key:1,drag:"",action:O,"on-success":X,"before-upload":A,"auto-upload":!0,limit:1,"on-error":q},{default:Vue.withCtx(()=>[Vue.createVNode(i,{slot:"trigger",type:"primary"},{default:Vue.withCtx(()=>[Vue.createTextVNode("选择文件")]),_:1}),ue,Vue.createTextVNode(" (最多上传一个文件) ")]),_:1})):Vue.createCommentVNode("",!0),d.value===2?(Vue.openBlock(),Vue.createBlock(T,{key:2},{default:Vue.withCtx(()=>[ae,Object.keys(V.value).length>0?(Vue.openBlock(),Vue.createBlock(L,{key:0,"label-width":"200"},{default:Vue.withCtx(()=>[Vue.createVNode(s,{justify:"center"},{default:Vue.withCtx(()=>[(Vue.openBlock(!0),Vue.createElementBlock(Vue.Fragment,null,Vue.renderList(V.value,(o,p)=>(Vue.openBlock(),Vue.createBlock(a,{key:p},{default:Vue.withCtx(()=>[Vue.createVNode(r,{label:p},{default:Vue.withCtx(()=>[Vue.createVNode(u,{modelValue:V.value[p],"onUpdate:modelValue":K=>V.value[p]=K,modelModifiers:{trim:!0},readonly:!0},null,8,["modelValue","onUpdate:modelValue"])]),_:2},1032,["label"])]),_:2},1024))),128))]),_:1})]),_:1})):Vue.createCommentVNode("",!0),Vue.createVNode(i,{onClick:z,type:"primary"},{default:Vue.withCtx(()=>[Vue.createTextVNode("返回上一步")]),_:1}),Vue.createVNode(i,{onClick:G,type:"primary",loading:v.value},{default:Vue.withCtx(()=>[Vue.createTextVNode("下载结果")]),_:1},8,["loading"]),Vue.createVNode(i,{type:"primary",size:"default",onClick:R},{default:Vue.withCtx(()=>[Vue.createTextVNode("下载转换前的特征")]),_:1}),Vue.createVNode(i,{type:"primary",size:"default",onClick:F},{default:Vue.withCtx(()=>[Vue.createTextVNode("下载转换后的特征")]),_:1})]),_:1})):Vue.createCommentVNode("",!0),d.value===3?(Vue.openBlock(),Vue.createBlock(T,{key:3},{default:Vue.withCtx(()=>[re,$.value.length>0?(Vue.openBlock(),Vue.createElementBlock("div",ne,[Vue.createElementVNode("ul",null,[(Vue.openBlock(!0),Vue.createElementBlock(Vue.Fragment,null,Vue.renderList($.value,(o,p)=>(Vue.openBlock(),Vue.createElementBlock("li",{key:p},Vue.toDisplayString(o),1))),128))])])):(Vue.openBlock(),Vue.createElementBlock("div",de,se))]),_:1})):Vue.createCommentVNode("",!0),d.value===1?(Vue.openBlock(),Vue.createBlock(i,{key:4,disabled:!(c.value.length>0),onClick:J,type:"primary",loading:f.value},{default:Vue.withCtx(()=>[Vue.createTextVNode("提交")]),_:1},8,["disabled","loading"])):Vue.createCommentVNode("",!0),Vue.withDirectives(Vue.createVNode(i,{onClick:z,slot:"tip"},{default:Vue.withCtx(()=>[Vue.createTextVNode("返回")]),_:1},512),[[Vue.vShow,d.value===1]])])}}});export{ce as default};
