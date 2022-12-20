from flask import Flask, jsonify, request
import pickle
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = pickle.load(open('loan.pickle', 'rb'))
scaler = pickle.load(open('loan_scaler.pickle', 'rb'))


@app.route("/")
@cross_origin()
def helloWorld():
    return 'Hello World'

@app.route("/loan", methods = ["GET"])
def predictLoan():
    if request.method == 'GET':
        a = float(request.args.get('loan_amnt'))
        b = float(request.args.get('term'))
        c = float(request.args.get('int_rate'))
        d = float(request.args.get('installment'))
        e = float(request.args.get('annual_inc'))
        f = float(request.args.get('dti'))
        g = float(request.args.get('open_acc'))
        h = float(request.args.get('pub_rec'))
        i = float(request.args.get('revol_bal'))
        j = float(request.args.get('revol_util'))
        k = float(request.args.get('total_acc'))
        l = float(request.args.get('mort_acc'))
        m = float(request.args.get('pub_rec_bankruptcies'))
        n = float(request.args.get('earliest_cr_year'))
        o = float(request.args.get('sub_grade_A2'))
        p = float(request.args.get('sub_grade_A3'))
        q = float(request.args.get('sub_grade_A4'))
        r = float(request.args.get('sub_grade_A5'))
        s = float(request.args.get('sub_grade_B1'))
        t = float(request.args.get('sub_grade_B2'))
        u = float(request.args.get('sub_grade_B3'))
        v = float(request.args.get('sub_grade_B4'))
        w = float(request.args.get('sub_grade_B5'))
        x = float(request.args.get('sub_grade_C1'))
        y = float(request.args.get('sub_grade_C2'))
        z = float(request.args.get('sub_grade_C3'))
        aa = float(request.args.get('sub_grade_C4'))
        bb = float(request.args.get('sub_grade_C5')) 
        cc = float(request.args.get('sub_grade_D1'))
        dd = float(request.args.get('sub_grade_D2'))
        ee = float(request.args.get('sub_grade_D3'))
        ff = float(request.args.get('sub_grade_D4'))
        gg = float(request.args.get('sub_grade_D5'))
        hh = float(request.args.get('sub_grade_E1'))
        ii = float(request.args.get('sub_grade_E2'))
        jj = float(request.args.get('sub_grade_E3'))
        kk = float(request.args.get('sub_grade_E4'))
        ll = float(request.args.get('sub_grade_E5'))
        mm = float(request.args.get('sub_grade_F1'))
        nn = float(request.args.get('sub_grade_F2'))
        oo = float(request.args.get('sub_grade_F3'))
        pp = float(request.args.get('sub_grade_F4'))
        qq = float(request.args.get('sub_grade_F5'))
        rr = float(request.args.get('sub_grade_G1'))
        ss = float(request.args.get('sub_grade_G2'))
        tt = float(request.args.get('sub_grade_G3'))
        uu = float(request.args.get('sub_grade_G4'))
        vv = float(request.args.get('sub_grade_G5'))
        ww = float(request.args.get('home_ownership_OTHER'))
        xx = float(request.args.get('home_ownership_OWN'))
        yy = float(request.args.get('home_ownership_RENT'))
        zz = float(request.args.get('verification_status_Source_Verified'))
        aaa = float(request.args.get('verification_status_income_Verified'))
        bbb = float(request.args.get('purpose_credit_card'))
        ccc = float(request.args.get('purpose_debt_consolidation'))
        ddd = float(request.args.get('purpose_educational'))
        eee = float(request.args.get('purpose_home_improvement'))
        fff = float(request.args.get('purpose_house'))
        ggg = float(request.args.get('purpose_major_purchase'))
        hhh = float(request.args.get('purpose_medical'))
        iii = float(request.args.get('purpose_moving'))
        jjj = float(request.args.get('purpose_other'))
        kkk = float(request.args.get('purpose_renewable_energy'))
        lll = float(request.args.get('purpose_small_business'))
        mmm = float(request.args.get('purpose_vacation'))
        nnn = float(request.args.get('purpose_wedding'))
        ooo = float(request.args.get('initial_list_status_w'))
        ppp = float(request.args.get('application_type_INDIVIDUAL'))
        qqq = float(request.args.get('application_type_JOINT'))
        rrr = float(request.args.get('zip_code_05113'))
        sss = float(request.args.get('zip_code_11650'))
        ttt = float(request.args.get('zip_code_22690'))
        uuu = float(request.args.get('zip_code_29597'))
        vvv = float(request.args.get('zip_code_30723'))
        www = float(request.args.get('zip_code_48052'))
        xxx = float(request.args.get('zip_code_70466'))
        yyy = float(request.args.get('zip_code_86630'))
        zzz = float(request.args.get('zip_code_93700'))
    
        final_features = ([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,
                                aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk,ll,mm,nn,oo,pp,qq,rr,ss,tt,uu,vv,ww,xx,yy,zz,
                                aaa,bbb,ccc,ddd,eee,fff,ggg,hhh,iii,jjj,kkk,lll,mmm,nnn,ooo,ppp,qqq,rrr,sss,ttt,
                                uuu,vvv,www,xxx,yyy,zzz]])

        final_features = scaler.transform(final_features)
    
        prediction = model.predict(final_features)
	
    return jsonify(str(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
