import{_ as l}from"./index-7cde0535.js";const m={class:"frame","element-loading-text":"加载中..."},d=["src"],V=Vue.defineComponent({name:"FrameView"}),v=Vue.defineComponent({...V,setup(p){var u,f,i;const r=Vue.ref(!0),t=VueRouter.useRoute(),o=Vue.ref(""),c=Vue.ref(null);(u=Vue.unref(t.meta))!=null&&u.frameSrc&&(o.value=(f=Vue.unref(t.meta))==null?void 0:f.frameSrc),((i=Vue.unref(t.meta))==null?void 0:i.frameLoading)===!1&&n();function n(){r.value=!1}function s(){Vue.nextTick(()=>{const e=Vue.unref(c);if(!e)return;const a=e;a.attachEvent?a.attachEvent("onload",()=>{n()}):e.onload=()=>{n()}})}return Vue.onMounted(()=>{s()}),(e,a)=>{const _=Vue.resolveDirective("loading");return Vue.withDirectives((Vue.openBlock(),Vue.createElementBlock("div",m,[Vue.createElementVNode("iframe",{src:o.value,class:"frame-iframe",ref_key:"frameRef",ref:c},null,8,d)])),[[_,r.value]])}}});const h=l(v,[["__scopeId","data-v-343b2dcc"]]);export{h as default};