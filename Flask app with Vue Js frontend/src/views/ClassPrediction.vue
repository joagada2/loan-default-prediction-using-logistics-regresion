<template>
  <div class="d-flex flex-column justify-content-center align-items-center">
    <div class="p-2">
      <h1 v-if="!APIResult.length">
        Enter Loan Application <br />Data to make prediction
      </h1>
      <h1 v-else style="font-size: 4rem">{{ APIResult }}</h1>
    </div>
    <div class="p-2">
      <form @submit.prevent>
        <div class="form-row" style="max-width: 500px">
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="a">Loan amount (in $)</label>
            <input
              class="form-control"
              v-model.trim="loandata.a"
              type="float"
              id="a"
            />
          </div>
          <div class="form-group col-md-3">
            <label for="b">Term</label>
            <select v-model.trim="loandata.b" class="form-control" id="b">
              <option value="36" selected>36</option>
              <option value="60" selected>60</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="c">Interest rate</label>
            <input
              class="form-control"
              v-model.trim="loandata.c"
              type="fload"
              id="c"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="d">Monthly installment ($)</label>
            <input
              class="form-control"
              v-model.trim="loandata.d"
              type="fload"
              id="d"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="e">Annual income ($)</label>
            <input
              class="form-control"
              v-model.trim="loandata.e"
              type="fload"
              id="e"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="f">DTI</label>
            <input
              class="form-control"
              v-model.trim="loandata.f"
              type="float"
              id="f"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="g"
              >Number of open credit lines</label
            >
            <input
              class="form-control"
              v-model.trim="loandata.g"
              type="float"
              id="g"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="h"
              >Number of deregetory Public record?</label
            >
            <input
              class="form-control"
              v-model.trim="loandata.h"
              type="float"
              id="h"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="i"
              >Total credit revolving balance ($)</label
            >
            <input
              class="form-control"
              v-model.trim="loandata.i"
              type="float"
              id="i"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="j"
              >Revolving line utilization rate</label
            >
            <input
              class="form-control"
              v-model.trim="loandata.j"
              type="float"
              id="j"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="k"
              >Number of current credit lines</label
            >
            <input
              class="form-control"
              v-model.trim="loandata.k"
              type="float"
              id="k"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="l"
              >Number of mortgage account(s)</label
            >
            <input
              class="form-control"
              v-model.trim="loandata.l"
              type="float"
              id="l"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="m"
              >Number of record of bankruptcy?</label
            >
            <input
              class="form-control"
              v-model.trim="loandata.m"
              type="float"
              id="m"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="n"
              >The year credit line was opened</label
            >
            <input
              class="form-control"
              v-model.trim="loandata.n"
              type="float"
              id="n"
            />
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="o">Subgrade A2?</label>
            <select v-model.trim="loandata.o" class="form-control" id="o">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="p">Subgrade A3?</label>
            <select v-model.trim="loandata.p" class="form-control" id="p">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="q">Subgrade A4?</label>
            <select v-model.trim="loandata.q" class="form-control" id="q">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="r">Subgrade A5?</label>
            <select v-model.trim="loandata.r" class="form-control" id="r">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="s">Subgrade B1?</label>
            <select v-model.trim="loandata.s" class="form-control" id="s">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="t">Subgrade B2?</label>
            <select v-model.trim="loandata.t" class="form-control" id="t">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="u">Subgrade B3?</label>
            <select v-model.trim="loandata.u" class="form-control" id="u">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="v">Subgrade B4?</label>
            <select v-model.trim="loandata.v" class="form-control" id="v">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="w">Subgrade B5?</label>
            <select v-model.trim="loandata.w" class="form-control" id="w">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="x">Subgrade C1?</label>
            <select v-model.trim="loandata.x" class="form-control" id="x">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="y">Subgrade C2?</label>
            <select v-model.trim="loandata.y" class="form-control" id="y">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="z">Subgrade C3?</label>
            <select v-model.trim="loandata.z" class="form-control" id="z">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="aa">Subgrade C4?</label>
            <select v-model.trim="loandata.aa" class="form-control" id="aa">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="bb">Subgrade C5?</label>
            <select v-model.trim="loandata.bb" class="form-control" id="bb">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="cc">Subgrade D1?</label>
            <select v-model.trim="loandata.cc" class="form-control" id="cc">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="dd">Subgrade D2?</label>
            <select v-model.trim="loandata.dd" class="form-control" id="dd">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="ee">Subgrade D3?</label>
            <select v-model.trim="loandata.ee" class="form-control" id="ee">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="ff">Subgrade D4?</label>
            <select v-model.trim="loandata.ff" class="form-control" id="ff">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="gg">Subgrade D5?</label>
            <select v-model.trim="loandata.gg" class="form-control" id="gg">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="hh">Subgrade E1?</label>
            <select v-model.trim="loandata.hh" class="form-control" id="hh">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="ii">Subgrade E2?</label>
            <select v-model.trim="loandata.ii" class="form-control" id="ii">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="jj">Subgrade E3?</label>
            <select v-model.trim="loandata.jj" class="form-control" id="jj">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="kk">Subgrade E4?</label>
            <select v-model.trim="loandata.kk" class="form-control" id="kk">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="ll">Subgrade E5?</label>
            <select v-model.trim="loandata.ll" class="form-control" id="ll">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="mm">Subgrade F1?</label>
            <select v-model.trim="loandata.mm" class="form-control" id="mm">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="nn">Subgrade F2?</label>
            <select v-model.trim="loandata.nn" class="form-control" id="nn">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="oo">Subgrade F3?</label>
            <select v-model.trim="loandata.oo" class="form-control" id="oo">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="pp">Subgrade F4?</label>
            <select v-model.trim="loandata.pp" class="form-control" id="pp">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="qq">Subgrade F5?</label>
            <select v-model.trim="loandata.qq" class="form-control" id="qq">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="rr">Subgrade G1?</label>
            <select v-model.trim="loandata.rr" class="form-control" id="rr">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="ss">Subgrade G2?</label>
            <select v-model.trim="loandata.ss" class="form-control" id="ss">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="tt">Subgrade G3?</label>
            <select v-model.trim="loandata.tt" class="form-control" id="tt">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="uu">Subgrade G4?</label>
            <select v-model.trim="loandata.uu" class="form-control" id="uu">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="vv">Subgrade G5?</label>
            <select v-model.trim="loandata.vv" class="form-control" id="vv">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="ww"
              >Other category of home ownership?</label
            >
            <select v-model.trim="loandata.ww" class="form-control" id="ww">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="xx">Own a home?</label>
            <select v-model.trim="loandata.xx" class="form-control" id="xx">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="yy">Renting a home?</label>
            <select v-model.trim="loandata.yy" class="form-control" id="yy">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="zz">Income source verified?</label>
            <select v-model.trim="loandata.zz" class="form-control" id="zz">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="aaa">Income amount verified?</label>
            <select v-model.trim="loandata.aaa" class="form-control" id="aaa">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="bbb">Loan for credit card?</label>
            <select v-model.trim="loandata.bbb" class="form-control" id="bbb">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="ccc"
              >Loan for Debt Consolidation?</label
            >
            <select v-model.trim="loandata.ccc" class="form-control" id="ccc">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="ddd">Loan for education?</label>
            <select v-model.trim="loandata.ddd" class="form-control" id="ddd">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="eee"
              >Loan for home improvement?</label
            >
            <select v-model.trim="loandata.eee" class="form-control" id="eee">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="fff">Loan for home purchase?</label>
            <select v-model.trim="loandata.fff" class="form-control" id="fff">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="ggg"
              >Loan for major purcahse?</label
            >
            <select v-model.trim="loandata.ggg" class="form-control" id="ggg">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="hhh">Loan for medicals?</label>
            <select v-model.trim="loandata.hhh" class="form-control" id="hhh">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="iii">Loan for moving?</label>
            <select v-model.trim="loandata.iii" class="form-control" id="iii">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="jjj"
              >Loan for other purchase?</label
            >
            <select v-model.trim="loandata.jjj" class="form-control" id="jjj">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="kkk"
              >Loan for renewable enegy?</label
            >
            <select v-model.trim="loandata.kkk" class="form-control" id="kkk">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="lll"
              >Loan for small business?</label
            >
            <select v-model.trim="loandata.lll" class="form-control" id="lll">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="mmm">Loan for vacation?</label>
            <select v-model.trim="loandata.mmm" class="form-control" id="mmm">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="nnn">Loan for wedding?</label>
            <select v-model.trim="loandata.nnn" class="form-control" id="nnn">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="ooo"
              >Is initial listing status W?</label
            >
            <select v-model.trim="loandata.ooo" class="form-control" id="ooo">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="ppp">Individual application?</label>
            <select v-model.trim="loandata.ppp" class="form-control" id="ppp">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="qqq">Joint application?</label>
            <select v-model.trim="loandata.qqq" class="form-control" id="qqq">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="rrr">Zip Code 05113?</label>
            <select v-model.trim="loandata.rrr" class="form-control" id="rrr">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="sss">Zip Code 11650?</label>
            <select v-model.trim="loandata.sss" class="form-control" id="sss">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="ttt">Zip Code 22690?</label>
            <select v-model.trim="loandata.ttt" class="form-control" id="ttt">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="uuu">Zip Code 29597?</label>
            <select v-model.trim="loandata.uuu" class="form-control" id="uuu">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="vvv">Zip Code 30723?</label>
            <select v-model.trim="loandata.vvv" class="form-control" id="vvv">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="www">Zip Code 48052?</label>
            <select v-model.trim="loandata.www" class="form-control" id="www">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="xxx">Zip Code 70466?</label>
            <select v-model.trim="loandata.xxx" class="form-control" id="xxx">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="yyy">Zip Code 86630?</label>
            <select v-model.trim="loandata.yyy" class="form-control" id="yyy">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <div class="form-group col-md-3">
            <label class="col-12 px-0" for="zzz">Zip Code 93700?</label>
            <select v-model.trim="loandata.zzz" class="form-control" id="zzz">
              <option value="1" selected>Yes</option>
              <option value="0" selected>No</option>
            </select>
          </div>
          <button
            @click="predict"
            type="button"
            class="btn btn-primary btn-lg btn-block"
          >
            Submit and scroll up to view your prediction
          </button>
        </div>
      </form>
    </div>
  </div>
</template>

<script>
import { getAPI } from "@/axios";
export default {
  name: "LoanData",
  data() {
    return {
      loandata: {
        a: "",
        b: "",
        c: "",
        d: "",
        e: "",
        f: "",
        g: "",
        h: "",
        i: "",
        j: "",
        k: "",
        l: "",
        m: "",
        n: "",
        o: "",
        p: "",
        q: "",
        r: "",
        s: "",
        t: "",
        u: "",
        v: "",
        w: "",
        x: "",
        y: "",
        z: "",
        aa: "",
        bb: "",
        cc: "",
        dd: "",
        ee: "",
        ff: "",
        gg: "",
        hh: "",
        ii: "",
        jj: "",
        kk: "",
        ll: "",
        mm: "",
        nn: "",
        oo: "",
        pp: "",
        qq: "",
        rr: "",
        ss: "",
        tt: "",
        uu: "",
        vv: "",
        ww: "",
        xx: "",
        yy: "",
        zz: "",
        aaa: "",
        bbb: "",
        ccc: "",
        ddd: "",
        eee: "",
        ffe: "",
        ggg: "",
        hhh: "",
        iii: "",
        jjj: "",
        kkk: "",
        lll: "",
        mmm: "",
        nnn: "",
        ooo: "",
        ppp: "",
        qqq: "",
        rrr: "",
        sss: "",
        ttt: "",
        uuu: "",
        vvv: "",
        www: "",
        xxx: "",
        yyy: "",
        zzz: "",
      },
      APIResult: [],
    };
  },
  methods: {
    predict() {
      getAPI
        .get("/loan", {
          params: {
            loan_amnt: this.loandata.a,
            term: this.loandata.b,
            int_rate: this.loandata.c,
            installment: this.loandata.d,
            annual_inc: this.loandata.e,
            dti: this.loandata.f,
            open_acc: this.loandata.g,
            pub_rec: this.loandata.h,
            revol_bal: this.loandata.i,
            revol_util: this.loandata.j,
            total_acc: this.loandata.k,
            mort_acc: this.loandata.l,
            pub_rec_bankruptcies: this.loandata.m,
            earliest_cr_year: this.loandata.n,
            sub_grade_A2: this.loandata.o,
            sub_grade_A3: this.loandata.p,
            sub_grade_A4: this.loandata.q,
            sub_grade_A5: this.loandata.r,
            sub_grade_B1: this.loandata.s,
            sub_grade_B2: this.loandata.t,
            sub_grade_B3: this.loandata.u,
            sub_grade_B4: this.loandata.v,
            sub_grade_B5: this.loandata.w,
            sub_grade_C1: this.loandata.x,
            sub_grade_C2: this.loandata.y,
            sub_grade_C3: this.loandata.z,
            sub_grade_C4: this.loandata.aa,
            sub_grade_C5: this.loandata.bb,
            sub_grade_D1: this.loandata.cc,
            sub_grade_D2: this.loandata.dd,
            sub_grade_D3: this.loandata.ee,
            sub_grade_D4: this.loandata.ff,
            sub_grade_D5: this.loandata.gg,
            sub_grade_E1: this.loandata.hh,
            sub_grade_E2: this.loandata.ii,
            sub_grade_E3: this.loandata.jj,
            sub_grade_E4: this.loandata.kk,
            sub_grade_E5: this.loandata.ll,
            sub_grade_F1: this.loandata.mm,
            sub_grade_F2: this.loandata.nn,
            sub_grade_F3: this.loandata.oo,
            sub_grade_F4: this.loandata.pp,
            sub_grade_F5: this.loandata.qq,
            sub_grade_G1: this.loandata.rr,
            sub_grade_G2: this.loandata.ss,
            sub_grade_G3: this.loandata.tt,
            sub_grade_G4: this.loandata.uu,
            sub_grade_G5: this.loandata.vv,
            home_ownership_OTHER: this.loandata.ww,
            home_ownership_OWN: this.loandata.xx,
            home_ownership_RENT: this.loandata.yy,
            verification_status_Source_Verified: this.loandata.zz,
            verification_status_income_Verified: this.loandata.aaa,
            purpose_credit_card: this.loandata.bbb,
            purpose_debt_consolidation: this.loandata.ccc,
            purpose_educational: this.loandata.ddd,
            purpose_home_improvement: this.loandata.eee,
            purpose_house: this.loandata.fff,
            purpose_major_purchase: this.loandata.ggg,
            purpose_medical: this.loandata.hhh,
            purpose_moving: this.loandata.iii,
            purpose_other: this.loandata.jjj,
            purpose_renewable_energy: this.loandata.kkk,
            purpose_small_business: this.loandata.lll,
            purpose_vacation: this.loandata.mmm,
            purpose_wedding: this.loandata.nnn,
            initial_list_status_w: this.loandata.ooo,
            application_type_INDIVIDUAL: this.loandata.ppp,
            application_type_JOINT: this.loandata.qqq,
            zip_code_05113: this.loandata.rrr,
            zip_code_11650: this.loandata.sss,
            zip_code_22690: this.loandata.ttt,
            zip_code_29597: this.loandata.uuu,
            zip_code_30723: this.loandata.vvv,
            zip_code_48052: this.loandata.www,
            zip_code_70466: this.loandata.xxx,
            zip_code_86630: this.loandata.yyy,
            zip_code_93700: this.loandata.zzz,
          },
        })
        .then((response) => {
          console.log("Recieved data successfully");
          const result = response.data;
          if (result == 0) {
            this.APIResult = "This Borrower WIll Default";
          } else {
            this.APIResult = "This Borrower WIll Not Default";
          }
          console.log(response.data);
        })
        .catch((err) => {
          console.log(err);
        });
    },
  },
};
</script>

<style></style>

