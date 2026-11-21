#pragma once
// BLDEPTH owns these fixed-parameter local formulas.  Both the standalone
// column and the full KPP translation-unit shim use this body.  Geometry,
// lookup tables and Python scalar parameters are fixed in this VJP slice.
#include <cmath>
#include <cstdint>

namespace locomgcm_bldepth {
#ifdef __CUDACC__
#define BL_LOCAL __host__ __device__ __forceinline__
#else
#define BL_LOCAL inline
#endif

template <typename T> BL_LOCAL T primal(T x) { return x; }
template <typename T> BL_LOCAL bool isnan(T x) { return std::isnan(x); }
template <typename T> BL_LOCAL T fabs(T x) { return ::fabs(x); }
template <typename T> BL_LOCAL T sqrt(T x) { return ::sqrt(x); }
template <typename T> BL_LOCAL T exp(T x) { return ::exp(x); }
template <typename T> BL_LOCAL T minimum_value(T a,T b) {return a<b?a:b;}
BL_LOCAL int64_t clamp_index(int64_t x,int64_t lo,int64_t hi) {return x<lo?lo:(x>hi?hi:x);}
template <typename scalar_t>
BL_LOCAL scalar_t clamp_min_nan(
    const scalar_t value,
    const scalar_t lower) {
  if (isnan(value)) {
    return value;
  }
  return value < lower ? lower : value;
}

template <typename scalar_t>
BL_LOCAL scalar_t minimum_nan(
    const scalar_t first,
    const scalar_t second) {
  if (isnan(first)) {
    return first;
  }
  if (isnan(second)) {
    return second;
  }
  return minimum_value(first, second);
}

template <typename scalar_t>
BL_LOCAL scalar_t fsign(
    const scalar_t magnitude,
    const scalar_t sign_source,
    const bool sign_zero_positive) {
  const scalar_t absolute = fabs(magnitude);
  if (sign_zero_positive) {
    return sign_source >= static_cast<scalar_t>(0.0) ? absolute : -absolute;
  }
  // Inspect the primal sign bit, including -0.0; SIGN's magnitude is fixed.
  return ::copysign(1.0, primal(sign_source)) < 0.0 ? -absolute : absolute;
}

template <typename scalar_t>
BL_LOCAL scalar_t swfrac(
    const scalar_t depth,
    const scalar_t rfac,
    const scalar_t one_minus_rfac,
    const scalar_t inv_a1,
    const scalar_t inv_a2) {
  const scalar_t facz = -depth;
  const scalar_t first = rfac * exp(facz * inv_a1);
  const scalar_t second = one_minus_rfac * exp(facz * inv_a2);
  const scalar_t fraction = first + second;
  return facz < -static_cast<scalar_t>(200.0) ? static_cast<scalar_t>(0.0) : fraction;
}

template <typename S> struct TableState {
  S value,wa,wb,ufrac,t00,t10,t01,t11;
};
template <typename scalar_t,typename table_t>
BL_LOCAL TableState<scalar_t> table_state(scalar_t zehat,scalar_t ustar,
    const table_t* wst,int64_t table_y,int64_t table_x,
    scalar_t inv_deltaz,scalar_t inv_deltau) {
  constexpr int64_t nni=890,nnj=480;
  const scalar_t zmin=-scalar_t(4.0e-7),umin=scalar_t(0);
  const scalar_t zdiff = zehat - zmin;
  const scalar_t iz_coordinate = zdiff * inv_deltaz;
  int64_t iz = static_cast<int64_t>(primal(iz_coordinate));
  iz = clamp_index(iz, 0, nni);
  const int64_t izp1 = iz + 1;
  const scalar_t udiff = ustar - umin;
  const scalar_t ju_coordinate = udiff * inv_deltau;
  int64_t ju = static_cast<int64_t>(primal(ju_coordinate));
  ju = clamp_index(ju, 0, nnj);
  const int64_t jup1 = ju + 1;
  const scalar_t zfrac_coordinate = zdiff * inv_deltaz;
  const scalar_t zfrac = zfrac_coordinate - static_cast<scalar_t>(iz);
  const scalar_t ufrac_coordinate = udiff * inv_deltau;
  const scalar_t ufrac = ufrac_coordinate - static_cast<scalar_t>(ju);
  const scalar_t fzfrac = static_cast<scalar_t>(1.0) - zfrac;

  const scalar_t t01=static_cast<scalar_t>(wst[iz*table_y+jup1*table_x]);
  const scalar_t t11=static_cast<scalar_t>(wst[izp1*table_y+jup1*table_x]);
  const scalar_t t00=static_cast<scalar_t>(wst[iz*table_y+ju*table_x]);
  const scalar_t t10=static_cast<scalar_t>(wst[izp1*table_y+ju*table_x]);
  const scalar_t wa_first = fzfrac * t01;
  const scalar_t wa_second = zfrac * t11;
  const scalar_t wa = wa_first + wa_second;
  const scalar_t wb_first = fzfrac * t00;
  const scalar_t wb_second = zfrac * t10;
  const scalar_t wb = wb_first + wb_second;
  const scalar_t table_first = (static_cast<scalar_t>(1.0) - ufrac) * wb;
  const scalar_t table_second = ufrac * wa;
  const scalar_t ws_table = table_first + table_second;

  return {ws_table,wa,wb,ufrac,t00,t10,t01,t11};
}
template <typename S> struct WscaleState {
  S value,zehat,first,second,u2,u3,numerator_first,numerator,denominator;
  TableState<S> lookup;
};
template <typename scalar_t, typename table_t>
BL_LOCAL WscaleState<scalar_t> wscale_state(
    const scalar_t sigma,
    const scalar_t hbl,
    const scalar_t ustar,
    const scalar_t bfsfc,
    const table_t* wst,
    const int64_t table_y, const int64_t table_x,
    const scalar_t inv_deltaz,
    const scalar_t inv_deltau) {
  const scalar_t zmax = static_cast<scalar_t>(0.0);
  const scalar_t wscale_vonk = static_cast<scalar_t>(0.4);
  const scalar_t conc1 = static_cast<scalar_t>(5.0);

  const scalar_t first = wscale_vonk * sigma;
  const scalar_t second = first * hbl;
  const scalar_t zehat = second * bfsfc;

  const auto lookup=table_state(zehat,ustar,wst,table_y,table_x,inv_deltaz,inv_deltau);
  const scalar_t ws_table=lookup.value;

  const scalar_t ustar_squared = ustar * ustar;
  const scalar_t ustar_cubed = ustar_squared * ustar;
  const scalar_t stable_numerator_first = wscale_vonk * ustar;
  const scalar_t stable_numerator = stable_numerator_first * ustar_cubed;
  const scalar_t stable_denominator_second = conc1 * zehat;
  const scalar_t stable_denominator = ustar_cubed + stable_denominator_second;
  const scalar_t ws_stable = stable_numerator /
      (zehat <= zmax ? static_cast<scalar_t>(1.0) : stable_denominator);
  return {zehat <= zmax ? ws_table : ws_stable,zehat,first,second,
      ustar_squared,ustar_cubed,stable_numerator_first,stable_numerator,
      stable_denominator,lookup};
}

template <typename S,typename T>
BL_LOCAL S wscale_ws(S sigma,S hbl,S ustar,S bfsfc,const T* wst,
    int64_t ty,int64_t tx,S inv_deltaz,S inv_deltau) {
  return wscale_state(sigma,hbl,ustar,bfsfc,wst,ty,tx,inv_deltaz,inv_deltau).value;
}

template <typename T> struct Constants {
  T ricr, epsilon, vtc, cekman, cmonob, min_hbl, phepsi;
  T rfac, one_minus_rfac, inv_a1, inv_a2, inv_deltaz, inv_deltau, inv_vonk;
  bool limit, penetrating_sw, sign_zero_positive;
};

// ufrac is shared by every level of a column.  Keep its two table-branch
// contributions separate until the column reduction reaches that node.
template <typename T> struct UstarTableSeeds {T wa=T(0),wb=T(0);};
template <typename T> struct WscaleSeeds {T ustar,bfsfc,sigma,hbl;};
template <typename T>
BL_LOCAL WscaleSeeds<T> wscale_ws_vjp(T sigma,T depth,T ustar,T bfsfc,
    const T* wst,int64_t ty,int64_t tx,const Constants<T>& c,T seed,
    UstarTableSeeds<T>* shared_ustar=nullptr) {
  const auto state=wscale_state(sigma,depth,ustar,bfsfc,wst,ty,tx,c.inv_deltaz,c.inv_deltau);
  T gu,gze;
  if(state.zehat<=T(0)) {
    const auto& q=state.lookup;
    // Reverse the shared four-corner DAG: aggregate the two zfrac and
    // two fzfrac nodes BEFORE multiplying by inv_deltaz.  Multiplying
    // each corner's forward partial first amplifies table cancellation.
    const T gwa=seed*q.ufrac,gwb=seed*(T(1)-q.ufrac);
    const T gufrac=seed*q.wa-seed*q.wb;
    const T gzfrac=gwb*q.t10+gwa*q.t11;
    const T gfzfrac=gwb*q.t00+gwa*q.t01;
    gze=(gzfrac-gfzfrac)*c.inv_deltaz;
    if(shared_ustar) {
      shared_ustar->wa += seed*q.wa;
      shared_ustar->wb += seed*q.wb;
      gu=T(0);
    } else gu=gufrac*c.inv_deltau;
  } else {
    const T u2=state.u2,u3=state.u3;
    const T numerator_first=state.numerator_first;
    const T numerator=state.numerator;
    const T denom=state.denominator;
    const T gn=seed/denom;
    const T gd=-seed*((numerator/denom)/denom);
    const T gu3=gd+gn*numerator_first;
    const T gu2=gu3*ustar;
    gu=gu3*u2+gu2*ustar+gu2*ustar+gn*u3*T(0.4);
    gze=gd*T(5);
  }
  return {gu,gze*state.second,gze*bfsfc*depth*T(0.4),gze*bfsfc*state.first};
}

template <typename S> struct RowState {
  S value,depth,absorbed,bfsfc,sigma,ws,dzprev,dznext,bvsq,root,product1,sum,denom,ritop;
};
template <typename S, typename T>
BL_LOCAL RowState<S> richardson_row_state(S dvsq,S dm,S dc,S ritop,S ustar,S bo,S bosol,
    T zp,T z,T zn,const T* wst,int64_t ty,int64_t tx,const Constants<T>& c) {
  const S depth=-S(z);
  const S penetrating=swfrac(depth,S(c.rfac),S(c.one_minus_rfac),S(c.inv_a1),S(c.inv_a2));
  const S absorbed=S(1)-penetrating;
  const S bfsfc_product=bosol*absorbed;
  const S bfsfc=c.penetrating_sw?bo+bfsfc_product:bo;
  const S stable=S(0.5)+fsign(S(0.5),bfsfc,c.sign_zero_positive);
  const S one_minus_stable=S(1)-stable;
  const S sigma_product=one_minus_stable*S(c.epsilon);
  const S sigma=stable+sigma_product;
  const S ws=wscale_ws(sigma,depth,ustar,bfsfc,wst,ty,tx,S(c.inv_deltaz),S(c.inv_deltau));
  const S dz_km1=S(zp)-S(z);
  const S dz_k=S(z)-S(zn);
  const S first=dm/dz_km1,second=dc/dz_k;
  const S bvsq=S(0.5)*(first+second);
  // Match Torch's safe sqrt on the bvsq == 0 branch.
  const S product1=depth*ws;
  const S root=sqrt(bvsq==S(0)?S(1):fabs(bvsq));
  S vtsq_branch=product1;
  vtsq_branch=vtsq_branch*root;
  vtsq_branch=vtsq_branch*S(c.vtc);
  const S vtsq=bvsq==S(0)?S(0):vtsq_branch;
  const S sum=dvsq+vtsq;
  const S denom=clamp_min_nan(sum,S(c.phepsi));
  return {ritop/denom,depth,absorbed,bfsfc,sigma,ws,dz_km1,dz_k,bvsq,root,product1,sum,denom,ritop};
}

template <typename S,typename T>
BL_LOCAL S richardson_row(S dvsq,S dm,S dc,S ritop,S ustar,S bo,S bosol,
    T zp,T z,T zn,const T* wst,int64_t ty,int64_t tx,const Constants<T>& c) {
  return richardson_row_state(dvsq,dm,dc,ritop,ustar,bo,bosol,zp,z,zn,wst,ty,tx,c).value;
}

template <typename T> struct RowSeeds {T dvsq,dm,dc,ritop,ustar,bo,bosol;};
template <typename T>
BL_LOCAL RowSeeds<T> richardson_row_vjp(T dvsq,T dm,T dc,T ritop,T ustar,T bo,T bosol,
    T zp,T z,T zn,const T* wst,int64_t ty,int64_t tx,const Constants<T>& c,T seed,
    UstarTableSeeds<T>* shared_ustar=nullptr) {
  const auto q=richardson_row_state(dvsq,dm,dc,ritop,ustar,bo,bosol,zp,z,zn,wst,ty,tx,c);
  const T gr=seed/q.denom;
  const T gden=-seed*((q.ritop/q.denom)/q.denom);
  const T gsum=q.sum>=c.phepsi?gden:T(0);
  const T gp=q.bvsq==T(0)?T(0):gsum*c.vtc;
  const T groot=gp*q.product1;
  const T gws=gp*q.root*q.depth;
  const T sign=q.bvsq>T(0)?T(1):(q.bvsq<T(0)?T(-1):T(0));
  const T gb=(groot/(T(2)*q.root))*sign;
  const auto w=wscale_ws_vjp(q.sigma,q.depth,ustar,q.bfsfc,wst,ty,tx,c,gws,shared_ustar);
  return {gsum,gb*T(0.5)/q.dzprev,gb*T(0.5)/q.dznext,gr,
      w.ustar,w.bfsfc,c.penetrating_sw?w.bfsfc*q.absorbed:T(0)};
}

template <typename T> struct FluxSeeds {T bo,bosol,depth;};
template <typename T>
BL_LOCAL FluxSeeds<T> clipped_flux_vjp(T depth,T bo,T bosol,const Constants<T>& c,T seed) {
  const T penetrating=swfrac(depth,c.rfac,c.one_minus_rfac,c.inv_a1,c.inv_a2);
  const T absorbed=T(1)-penetrating;
  const T product=bosol*absorbed;
  const T raw=c.penetrating_sw?bo+product:bo;
  const T signed_seed=seed*fsign(T(1),raw,c.sign_zero_positive);
  const T magnitude_seed=fabs(raw)>=c.phepsi?signed_seed:T(0);
  const T raw_seed=magnitude_seed*(raw>T(0)?T(1):(raw<T(0)?T(-1):T(0)));
  if(!c.penetrating_sw) return {raw_seed,T(0),T(0)};
  const T solar_seed=raw_seed*absorbed;
  const T fraction_seed=-(raw_seed*bosol);
  const T facz=-depth;
  const T selected_seed=facz<T(-200)?T(0):fraction_seed;
  const T first_exp=exp(facz*c.inv_a1),second_exp=exp(facz*c.inv_a2);
  const T first_seed=((selected_seed*c.rfac)*first_exp)*c.inv_a1;
  const T second_seed=((selected_seed*c.one_minus_rfac)*second_exp)*c.inv_a2;
  return {raw_seed,solar_seed,-(first_seed+second_seed)};
}

// Seeded reverse of the column tail.  Applying the seed at each reverse node
// preserves the floating-point order of the reference division pullback;
// forming local forward partials and multiplying the seed afterwards does not.
template <typename T> struct TailSeeds {T rk,rprev,ustar,bo,bosol,coriol;};
template <typename T>
BL_LOCAL TailSeeds<T> column_tail_vjp(int64_t kmt,int64_t kb,T rk,T rprev,
    T ustar,T bo,T bosol,T coriol,const T* zg,int64_t zs,int nz,
    const Constants<T>& c,T hbl_seed,T bfsfc_seed) {
  const int64_t initial=clamp_index(kmt<1?1:kmt,0,nz+1);
  const T zk=zg[clamp_index(kb,0,nz+1)*zs],zp=zg[clamp_index(kb-1,0,nz+1)*zs];
  const T denom_raw=rk-rprev,denom=denom_raw==T(0)?T(1):denom_raw;
  const T dz=zp-zk,product=dz*(c.ricr-rprev);
  const bool interpolate=kb>1 && kb<kmt;
  const T hpre=interpolate?-zp+product/denom:-zg[initial*zs];
  const T absorbed=T(1)-swfrac(hpre,c.rfac,c.one_minus_rfac,c.inv_a1,c.inv_a2);
  const T solar=bosol*absorbed;
  const T raw=c.penetrating_sw?bo+solar:bo;
  const T stable=T(0.5)+fsign(T(0.5),raw,c.sign_zero_positive);
  const T clipped=fsign(T(1),raw,c.sign_zero_positive)*clamp_min_nan(fabs(raw),c.phepsi);
  const T fden=clamp_min_nan(fabs(coriol),c.phepsi);
  const T enum_first=c.cekman*ustar,ekman=enum_first/fden;
  const T m1=c.cmonob*ustar,m2=m1*ustar,m3=m2*ustar;
  const T mnum=m3*c.inv_vonk,monob=mnum/clipped;
  const T minimum_limit=minimum_nan(ekman,monob);
  const T hlimit=stable*minimum_limit+(stable-T(1))*zg[nz*zs];
  const bool limited=c.limit && clipped>T(0);
  const T hbefore=limited?minimum_nan(hpre,hlimit):hpre;
  const T hfinal=clamp_min_nan(hbefore,c.min_hbl);
  const auto final_flux=clipped_flux_vjp(hfinal,bo,bosol,c,bfsfc_seed);
  const T htotal=hbl_seed+final_flux.depth;
  const T gh=hbefore>=c.min_hbl?htotal:T(0);
  T gpre=gh,gu=T(0),gf=T(0),gbo=final_flux.bo,gsol=final_flux.bosol;
  if(limited) {
    gpre=hpre>hlimit?T(0):(hpre==hlimit?gh*T(0.5):gh);
    const T glimit=hlimit>hpre?T(0):(hpre==hlimit?gh*T(0.5):gh);
    const T gminimum=glimit*stable;
    const T gek=ekman>monob?T(0):(ekman==monob?gminimum*T(0.5):gminimum);
    const T gmo=monob>ekman?T(0):(ekman==monob?gminimum*T(0.5):gminimum);
    const T genum=gek/fden,gfden=-gek*((enum_first/fden)/fden);
    const T gabsf=fabs(coriol)>=c.phepsi?gfden:T(0);
    gf=gabsf*(coriol>T(0)?T(1):(coriol<T(0)?T(-1):T(0)));
    const T gmnum=gmo/clipped,gclipped=-gmo*((mnum/clipped)/clipped);
    const T gm3=gmnum*c.inv_vonk,gm2=gm3*ustar,gm1=gm2*ustar;
    gu=gm3*m2+gm2*m1+gm1*c.cmonob+genum*c.cekman;
    const auto initial_flux=clipped_flux_vjp(hpre,bo,bosol,c,gclipped);
    gpre=gpre+initial_flux.depth;
    gbo=gbo+initial_flux.bo;gsol=gsol+initial_flux.bosol;
  }
  T grk=T(0),grp=T(0);
  if(interpolate) {
    const T gproduct=gpre/denom;
    const T gdenom=-gpre*((product/denom)/denom);
    grk=denom_raw==T(0)?T(0):gdenom;
    grp=-(gproduct*dz)-grk;
  }
  return {grk,grp,gu,gbo,gsol,gf};
}
template <typename T>
inline Constants<T> constants(double ricr,double epsilon,double vtc,double cekman,
    double cmonob,double min_kpp_hbl,double vonk,double phepsi,bool limit,int jwtype,
    bool penetrating_sw,bool sign_zero_positive) {
  constexpr int64_t nni = 890;
  constexpr int64_t nnj = 480;
  // PyTorch CUDA div.Scalar computes these reciprocals on the host and then
  // launches multiplication. Preserve that lowering instead of device divide.
  const double deltaz = (0.0 - (-4.0e-7)) / static_cast<double>(nni + 1);
  const double deltau = (4.0e-2 - 0.0) / static_cast<double>(nnj + 1);
  const double inv_deltaz = 1.0 / deltaz;
  const double inv_deltau = 1.0 / deltau;
  double swfrac_rfac = 0.62;
  double swfrac_a1 = 0.6;
  double swfrac_a2 = 20.0;
  if (jwtype == 1) {
    swfrac_rfac = 0.58;
    swfrac_a1 = 0.35;
    swfrac_a2 = 23.0;
  } else if (jwtype == 3) {
    swfrac_rfac = 0.67;
    swfrac_a1 = 1.0;
    swfrac_a2 = 17.0;
  } else if (jwtype == 4) {
    swfrac_rfac = 0.77;
    swfrac_a1 = 1.5;
    swfrac_a2 = 14.0;
  } else if (jwtype == 5) {
    swfrac_rfac = 0.78;
    swfrac_a1 = 1.4;
    swfrac_a2 = 7.9;
  }
  const double swfrac_one_minus_rfac = 1.0 - swfrac_rfac;
  const double swfrac_inv_a1 = 1.0 / swfrac_a1;
  const double swfrac_inv_a2 = 1.0 / swfrac_a2;
  const double inv_vonk = 1.0 / vonk;
  return {T(ricr),T(epsilon),T(vtc),T(cekman),T(cmonob),T(min_kpp_hbl),T(phepsi),
      T(swfrac_rfac),T(swfrac_one_minus_rfac),T(swfrac_inv_a1),T(swfrac_inv_a2),
      T(inv_deltaz),T(inv_deltau),T(inv_vonk),limit,penetrating_sw,sign_zero_positive};
}

#undef BL_LOCAL
} // namespace locomgcm_bldepth
