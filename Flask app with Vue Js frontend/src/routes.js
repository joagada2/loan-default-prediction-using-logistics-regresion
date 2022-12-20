import Vue from "vue";
import VueRouter from "vue-router";
import ClassPrediction from "./views/ClassPrediction";

Vue.use(VueRouter);

export default new VueRouter({
  mode: "history",
  routes: [
    {
      path: "/",
      name: "prediction",
      component: ClassPrediction,
    },
  ],
});
