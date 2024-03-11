export default {
  path: "/CGN",
  redirect: "/CGN/List",
  meta: {
    icon: "twemoji:man-scientist-dark-skin-tone",
    title: "实验",
    // showLink: false,
    rank: 1,
    
  },
  children: [
    {
      path: "/CGN/feature_set_choose",
      name: "feature_set_choose",
      showParent: true,
      component: () => import("@/views/CGN/feature_set_choose.vue"),
      meta: {
        title: "特征子集选择",
        keepAlive:true
      }
    },
    {
      path: "/CGN/finance_symbolic_regression",
      name: "finance_symbolic_regression",
      showParent: true,
      component: () => import("@/views/CGN/finance_symbolic_regression.vue"),
      meta: {
        title: "符号回归构建公式",
        keepAlive:true
      }
    },
    {
      path: "/CGN/symbolic_regression",
      name: "symbolic_regression",
      showParent: true,
      component: () => import("@/views/CGN/symbolic_regression.vue"),
      meta: {
        title: "符号回归构建特征",
        keepAlive:true
      }
    },
    {
      path: "/CGN/ANN_model_train",
      name: "ANN_model_train",
      showParent: true,
      component: () => import("@/views/CGN/ANN_model_train.vue"),
      meta: {
        title: "ANN模型训练",
        keepAlive:true
      }
    },
    {
      path: "/CGN/ANN_model_predict",
      name: "ANN_model_predict",
      showParent: true,
      component: () => import("@/views/CGN/ANN_model_predict.vue"),
      meta: {
        title: "ANN模型预测",
        keepAlive:true
      }
    },
  ]
} as unknown as RouteConfigsTable;
