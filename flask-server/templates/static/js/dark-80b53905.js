import{O as A,F as U,w as W,m as i,g as d,E as J,s as P,G as q,I as E,H as K,u as X,J as L,M as Y,p as H,N as Z}from"./index-2c723bdb.js";function ee(){const{$storage:t,$config:e}=A(),o=()=>{U().multiTagsCache&&(!t.tags||t.tags.length===0)&&(t.tags=W),t.layout||(t.layout={layout:(e==null?void 0:e.Layout)??"vertical",theme:(e==null?void 0:e.Theme)??"default",darkMode:(e==null?void 0:e.DarkMode)??!1,sidebarStatus:(e==null?void 0:e.SidebarStatus)??!0,epThemeColor:(e==null?void 0:e.EpThemeColor)??"#409EFF"}),t.configure||(t.configure={grey:(e==null?void 0:e.Grey)??!1,weak:(e==null?void 0:e.Weak)??!1,hideTabs:(e==null?void 0:e.HideTabs)??!1,showLogo:(e==null?void 0:e.ShowLogo)??!0,showModel:(e==null?void 0:e.ShowModel)??"smart",multiTagsCache:(e==null?void 0:e.MultiTagsCache)??!1})},n=Vue.computed(()=>t==null?void 0:t.layout.layout),a=Vue.computed(()=>t.layout);return{layout:n,layoutTheme:a,initStorage:o}}const te=Pinia.defineStore({id:"pure-app",state:()=>{var t,e;return{sidebar:{opened:((t=i().getItem("responsive-layout"))==null?void 0:t.sidebarStatus)??d().SidebarStatus,withoutAnimation:!1,isClickCollapse:!1},layout:((e=i().getItem("responsive-layout"))==null?void 0:e.layout)??d().Layout,device:J()?"mobile":"desktop"}},getters:{getSidebarStatus(){return this.sidebar.opened},getDevice(){return this.device}},actions:{TOGGLE_SIDEBAR(t,e){const o=i().getItem("responsive-layout");t&&e?(this.sidebar.withoutAnimation=!0,this.sidebar.opened=!0,o.sidebarStatus=!0):!t&&e?(this.sidebar.withoutAnimation=!0,this.sidebar.opened=!1,o.sidebarStatus=!1):!t&&!e&&(this.sidebar.withoutAnimation=!1,this.sidebar.opened=!this.sidebar.opened,this.sidebar.isClickCollapse=!this.sidebar.opened,o.sidebarStatus=this.sidebar.opened),i().setItem("responsive-layout",o)},async toggleSideBar(t,e){await this.TOGGLE_SIDEBAR(t,e)},toggleDevice(t){this.device=t},setLayout(t){this.layout=t}}});function ne(){return te(P)}const oe=Pinia.defineStore({id:"pure-epTheme",state:()=>{var t,e;return{epThemeColor:((t=i().getItem("responsive-layout"))==null?void 0:t.epThemeColor)??d().EpThemeColor,epTheme:((e=i().getItem("responsive-layout"))==null?void 0:e.theme)??d().Theme}},getters:{getEpThemeColor(){return this.epThemeColor},fill(){return this.epTheme==="light"?"#409eff":this.epTheme==="yellow"?"#d25f00":"#fff"}},actions:{setEpThemeColor(t){const e=i().getItem("responsive-layout");this.epTheme=e==null?void 0:e.theme,this.epThemeColor=t,e&&(e.epThemeColor=t,i().setItem("responsive-layout",e))}}});function S(){return oe(P)}function $e(t,e){const o=/^IF-/;if(o.test(t)){const n=t.split(o)[1],a=n.slice(0,n.indexOf(" ")==-1?n.length:n.indexOf(" ")),s=n.slice(n.indexOf(" ")+1,n.length);return Vue.defineComponent({name:"FontIcon",render(){return Vue.h(q,{icon:a,iconType:s,...e})}})}else return typeof t=="function"||typeof(t==null?void 0:t.render)=="function"?t:typeof t=="object"?Vue.defineComponent({name:"OfflineIcon",render(){return Vue.h(E,{icon:t,...e})}}):Vue.defineComponent({name:"Icon",render(){const n=t&&t.includes(":")?K:E;return Vue.h(n,{icon:t,...e})}})}const w="当前路由配置不正确，请检查配置";function ye(){var I;const t=VueRouter.useRoute(),e=ne(),o=VueRouter.useRouter().options.routes,{wholeMenus:n}=Pinia.storeToRefs(X()),a=((I=d())==null?void 0:I.TooltipEffect)??"light",s=Vue.computed(()=>{var u;return(u=L())==null?void 0:u.username}),g=Vue.computed(()=>s.value?{marginRight:"10px"}:""),b=Vue.computed(()=>!e.getSidebarStatus),v=Vue.computed(()=>e.getDevice),{$storage:c,$config:$}=A(),l=Vue.computed(()=>{var u;return(u=c==null?void 0:c.layout)==null?void 0:u.layout}),r=Vue.computed(()=>$.Title);function f(u){const m=d().Title;m?document.title=`${u.title} | ${m}`:document.title=u.title}function y(){L().logOut()}function C(){Y.push("/welcome")}function z(){H.emit("openPanel")}function D(){e.toggleSideBar()}function G(u){u==null||u.handleResize()}function j(u){var T;if(!u.children)return console.error(w);const m=/^http(s?):\/\//,h=(T=u.children[0])==null?void 0:T.path;return m.test(h)?u.path+"/"+h:h}function F(u,m){if(n.value.length===0||Q(u))return;let h="";const T=u.lastIndexOf("/");T>0&&(h=u.slice(0,T));function M(B,x){return x?x.map(p=>{p.path===B?p.redirect?M(p.redirect,p.children):H.emit("changLayoutRoute",{indexPath:B,parentPath:h}):p.children&&M(B,p.children)}):console.error(w)}M(u,m)}function Q(u){return Z.includes(u)}return{route:t,title:r,device:v,layout:l,logout:y,routers:o,$storage:c,backHome:C,onPanel:z,changeTitle:f,toggleSideBar:D,menuSelect:F,handleResize:G,resolvePath:j,isCollapse:b,pureApp:e,username:s,avatarsStyle:g,tooltipEffect:a}}const k={outputDir:"",defaultScopeName:"",includeStyleWithColors:[],extract:!0,themeLinkTagId:"theme-link-tag",themeLinkTagInjectTo:"head",removeCssScopeName:!1,customThemeCssFileName:null,arbitraryMode:!1,defaultPrimaryColor:"",customThemeOutputPath:"C:/Users/83985/Desktop/github/cp/cp/admin/node_modules/.pnpm/@pureadmin+theme@3.0.0/node_modules/@pureadmin/theme/setCustomTheme.js",styleTagId:"custom-theme-tagid",InjectDefaultStyleTagToHtml:!0,hueDiffControls:{low:0,high:0},multipleScopeVars:[{scopeName:"layout-theme-default",varsContent:`
        $subMenuActiveText: #fff !default;
        $menuBg: #001529 !default;
        $menuHover: #4091f7 !default;
        $subMenuBg: #0f0303 !default;
        $subMenuActiveBg: #4091f7 !default;
        $menuText: rgb(254 254 254 / 65%) !default;
        $sidebarLogo: #002140 !default;
        $menuTitleHover: #fff !default;
        $menuActiveBefore: #4091f7 !default;
      `},{scopeName:"layout-theme-light",varsContent:`
        $subMenuActiveText: #409eff !default;
        $menuBg: #fff !default;
        $menuHover: #e0ebf6 !default;
        $subMenuBg: #fff !default;
        $subMenuActiveBg: #e0ebf6 !default;
        $menuText: #7a80b4 !default;
        $sidebarLogo: #fff !default;
        $menuTitleHover: #000 !default;
        $menuActiveBefore: #4091f7 !default;
      `},{scopeName:"layout-theme-dusk",varsContent:`
        $subMenuActiveText: #fff !default;
        $menuBg: #2a0608 !default;
        $menuHover: #e13c39 !default;
        $subMenuBg: #000 !default;
        $subMenuActiveBg: #e13c39 !default;
        $menuText: rgb(254 254 254 / 65.1%) !default;
        $sidebarLogo: #42090c !default;
        $menuTitleHover: #fff !default;
        $menuActiveBefore: #e13c39 !default;
      `},{scopeName:"layout-theme-volcano",varsContent:`
        $subMenuActiveText: #fff !default;
        $menuBg: #2b0e05 !default;
        $menuHover: #e85f33 !default;
        $subMenuBg: #0f0603 !default;
        $subMenuActiveBg: #e85f33 !default;
        $menuText: rgb(254 254 254 / 65%) !default;
        $sidebarLogo: #441708 !default;
        $menuTitleHover: #fff !default;
        $menuActiveBefore: #e85f33 !default;
      `},{scopeName:"layout-theme-yellow",varsContent:`
        $subMenuActiveText: #d25f00 !default;
        $menuBg: #2b2503 !default;
        $menuHover: #f6da4d !default;
        $subMenuBg: #0f0603 !default;
        $subMenuActiveBg: #f6da4d !default;
        $menuText: rgb(254 254 254 / 65%) !default;
        $sidebarLogo: #443b05 !default;
        $menuTitleHover: #fff !default;
        $menuActiveBefore: #f6da4d !default;
      `},{scopeName:"layout-theme-mingQing",varsContent:`
        $subMenuActiveText: #fff !default;
        $menuBg: #032121 !default;
        $menuHover: #59bfc1 !default;
        $subMenuBg: #000 !default;
        $subMenuActiveBg: #59bfc1 !default;
        $menuText: #7a80b4 !default;
        $sidebarLogo: #053434 !default;
        $menuTitleHover: #fff !default;
        $menuActiveBefore: #59bfc1 !default;
      `},{scopeName:"layout-theme-auroraGreen",varsContent:`
        $subMenuActiveText: #fff !default;
        $menuBg: #0b1e15 !default;
        $menuHover: #60ac80 !default;
        $subMenuBg: #000 !default;
        $subMenuActiveBg: #60ac80 !default;
        $menuText: #7a80b4 !default;
        $sidebarLogo: #112f21 !default;
        $menuTitleHover: #fff !default;
        $menuActiveBefore: #60ac80 !default;
      `},{scopeName:"layout-theme-pink",varsContent:`
        $subMenuActiveText: #fff !default;
        $menuBg: #28081a !default;
        $menuHover: #d84493 !default;
        $subMenuBg: #000 !default;
        $subMenuActiveBg: #d84493 !default;
        $menuText: #7a80b4 !default;
        $sidebarLogo: #3f0d29 !default;
        $menuTitleHover: #fff !default;
        $menuActiveBefore: #d84493 !default;
      `},{scopeName:"layout-theme-saucePurple",varsContent:`
        $subMenuActiveText: #fff !default;
        $menuBg: #130824 !default;
        $menuHover: #693ac9 !default;
        $subMenuBg: #000 !default;
        $subMenuActiveBg: #693ac9 !default;
        $menuText: #7a80b4 !default;
        $sidebarLogo: #1f0c38 !default;
        $menuTitleHover: #fff !default;
        $menuActiveBefore: #693ac9 !default;
      `}]},ue="/",ae="assets";function R(t){let e=t.replace("#","").match(/../g);for(let o=0;o<3;o++)e[o]=parseInt(e[o],16);return e}function O(t,e,o){let n=[t.toString(16),e.toString(16),o.toString(16)];for(let a=0;a<3;a++)n[a].length==1&&(n[a]=`0${n[a]}`);return`#${n.join("")}`}function re(t,e){let o=R(t);for(let n=0;n<3;n++)o[n]=Math.floor(o[n]*(1-e));return O(o[0],o[1],o[2])}function le(t,e){let o=R(t);for(let n=0;n<3;n++)o[n]=Math.floor((255-o[n])*e+o[n]);return O(o[0],o[1],o[2])}function V(t){return`(^${t}\\s+|\\s+${t}\\s+|\\s+${t}$|^${t}$)`}function N({scopeName:t,multipleScopeVars:e}){const o=Array.isArray(e)&&e.length?e:k.multipleScopeVars;let n=document.documentElement.className;new RegExp(V(t)).test(n)||(o.forEach(a=>{n=n.replace(new RegExp(V(a.scopeName),"g"),` ${t} `)}),document.documentElement.className=n.replace(/(^\s+|\s+$)/g,""))}function _({id:t,href:e}){const o=document.createElement("link");return o.rel="stylesheet",o.href=e,o.id=t,o}function se(t){const e={scopeName:"theme-default",customLinkHref:s=>s,...t},o=e.themeLinkTagId||k.themeLinkTagId;let n=document.getElementById(o);const a=e.customLinkHref(`${ue.replace(/\/$/,"")}${`/${ae}/${e.scopeName}.css`.replace(/\/+(?=\/)/g,"")}`);if(n){n.id=`${o}_old`;const s=_({id:o,href:a});n.nextSibling?n.parentNode.insertBefore(s,n.nextSibling):n.parentNode.appendChild(s),s.onload=()=>{setTimeout(()=>{n.parentNode.removeChild(n),n=null},60),N(e)};return}n=_({id:o,href:a}),N(e),document[(e.themeLinkTagInjectTo||k.themeLinkTagInjectTo||"").replace("-prepend","")].appendChild(n)}function Ce(){var $;const{layoutTheme:t,layout:e}=ee(),o=Vue.ref([{color:"#1b2a47",themeColor:"default"},{color:"#ffffff",themeColor:"light"},{color:"#f5222d",themeColor:"dusk"},{color:"#fa541c",themeColor:"volcano"},{color:"#fadb14",themeColor:"yellow"},{color:"#13c2c2",themeColor:"mingQing"},{color:"#52c41a",themeColor:"auroraGreen"},{color:"#eb2f96",themeColor:"pink"},{color:"#722ed1",themeColor:"saucePurple"}]),{$storage:n}=A(),a=Vue.ref(($=n==null?void 0:n.layout)==null?void 0:$.darkMode),s=document.documentElement;function g(l="default"){var r,f;if(t.value.theme=l,se({scopeName:`layout-theme-${l}`}),n.layout={layout:e.value,theme:l,darkMode:a.value,sidebarStatus:(r=n.layout)==null?void 0:r.sidebarStatus,epThemeColor:(f=n.layout)==null?void 0:f.epThemeColor},l==="default"||l==="light")v(d().EpThemeColor);else{const y=o.value.find(C=>C.themeColor===l);v(y.color)}}function b(l,r,f){document.documentElement.style.setProperty(`--el-color-primary-${l}-${r}`,a.value?re(f,r/10):le(f,r/10))}const v=l=>{S().setEpThemeColor(l),document.documentElement.style.setProperty("--el-color-primary",l);for(let r=1;r<=2;r++)b("dark",r,l);for(let r=1;r<=9;r++)b("light",r,l)};function c(){S().epTheme==="light"&&a.value?g("default"):g(S().epTheme),a.value?document.documentElement.classList.add("dark"):document.documentElement.classList.remove("dark")}return{body:s,dataTheme:a,layoutTheme:t,themeColors:o,dataThemeChange:c,setEpThemeColor:v,setLayoutThemeColor:g}}const ie={xmlns:"http://www.w3.org/2000/svg",width:"16",height:"16",viewBox:"0 0 24 24"},fe=Vue.createElementVNode("path",{fill:"none",d:"M0 0h24v24H0z"},null,-1),de=Vue.createElementVNode("path",{d:"M12 18a6 6 0 1 1 0-12 6 6 0 0 1 0 12zM11 1h2v3h-2V1zm0 19h2v3h-2v-3zM3.515 4.929l1.414-1.414L7.05 5.636 5.636 7.05 3.515 4.93zM16.95 18.364l1.414-1.414 2.121 2.121-1.414 1.414-2.121-2.121zm2.121-14.85 1.414 1.415-2.121 2.121-1.414-1.414 2.121-2.121zM5.636 16.95l1.414 1.414-2.121 2.121-1.414-1.414 2.121-2.121zM23 11v2h-3v-2h3zM4 11v2H1v-2h3z"},null,-1),ce=[fe,de];function me(t,e){return Vue.openBlock(),Vue.createElementBlock("svg",ie,ce)}const Me={render:me},he={xmlns:"http://www.w3.org/2000/svg",width:"16",height:"16",viewBox:"0 0 24 24"},pe=Vue.createElementVNode("path",{fill:"none",d:"M0 0h24v24H0z"},null,-1),ge=Vue.createElementVNode("path",{d:"M11.38 2.019a7.5 7.5 0 1 0 10.6 10.6C21.662 17.854 17.316 22 12.001 22 6.477 22 2 17.523 2 12c0-5.315 4.146-9.661 9.38-9.981z"},null,-1),ve=[pe,ge];function Te(t,e){return Vue.openBlock(),Vue.createElementBlock("svg",he,ve)}const Be={render:Te};export{$e as a,ye as b,Ce as c,ne as d,Me as e,Be as f,ee as g,se as t,S as u};
