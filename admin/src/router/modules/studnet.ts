export default {
  path: "/student",
  redirect: "/student/List",
  meta: {
    icon: "twemoji:man-scientist-dark-skin-tone",
    title: "实验",
    // showLink: false,
    rank: 1,
  },
  children: [
    {
      path: "/student/add",
      name: "studentAdd",
      showParent: true,
      component: () => import("@/views/student/studentAdd.vue"),
      meta: {
        title: "符号回归特征"
      }
    },
    {
      path: "/student/list",
      name: "studentList",
      showParent: true,
      component: () => import("@/views/student/studentList.vue"),
      meta: {
        title: "特征选择"
      }
    },
    {
      path: "/student/t",
      name: "studentt",
      showParent: true,
      component: () => import("@/views/student/studentt.vue"),
      meta: {
        title: "训练模型"
      }
    },
  ]
} as unknown as RouteConfigsTable;
