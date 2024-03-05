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
        title: "特征子集选择"
      }
    },
    {
      path: "/CGN/symbolic_regression",
      name: "symbolic_regression",
      showParent: true,
      component: () => import("@/views/CGN/symbolic_regression.vue"),
      meta: {
        title: "符号回归特征"
      }
    },
    {
      path: "/CGN/ANN_model_train",
      name: "ANN_model_train",
      showParent: true,
      component: () => import("@/views/CGN/ANN_model_train.vue"),
      meta: {
        title: "训练模型"
      }
    },
  ]
} as unknown as RouteConfigsTable;
