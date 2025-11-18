# Thin lens equation for a real laser beam with weak lens aperture truncation

Haiyin Sun, MEMBER SPIE

Coherent, Inc.

Auburn Group

2303 Lindbergh Street

M/S A90

Auburn, California 95602

E-mail: haiyin_sun@cohr.com

Abstract. The effects of a thin lens on a real laser beam with an  $M^2$  factor larger than 1 is theoretically studied. The effect of weak lens aperture truncation is considered by introducing an increment to the  $M^2$  factor. A new thin lens equation is derived. This equation is a simple and effective tool applicable to real laser beams, and can be used to estimate the phenomena of focal shift and change in focused spot size caused by weak lens aperture truncation. Two examples of applications are described. © 1998 Society of Photo-Optical Instrumentation Engineers. [S0091-3286(98)01911-4]

Subject terms: laser beams; thin lens; aperture truncation.

Paper 980033 received Jan. 26, 1998; revised manuscript received June 18, 1998; accepted for publication Aug. 3, 1998.

# 1 Introduction

Lasers have been widely used since their invention. Laser beams can be approximated by a basic spatial mode ideal (BSMI) Gaussian beam. In most applications, laser beams are manipulated by lenses or mirrors. How a lens manipulates a BSMI Gaussian beam has been studied. $^{1-8}$  It has also been shown that lens aperture truncation on a BSMI Gaussian beam can change the beam characteristics. This includes shifting the beam focal point and changing the beam focused spot size and divergence. These changes in beam characteristics are determined by the intensity level of the lens aperture truncation on the beam. It has also been shown that a BSMI Gaussian beam truncated by an aperture at a low intensity level can still be approximated by another BSMI Gaussian beam with different beam waist. $^{7}$  In all these studies, only a BSMI Gaussian beam was considered, and diffraction optics theory $^{1-7}$  or ray-equivalent modeling $^{8}$  were used. To simplify the complex calculations involved in diffraction optics theory, approximations were often used. For example, the lens aperture was sometimes assumed to be much smaller than the distance from the point of interest to the lens aperture. The results obtained were numerical and complex.

On the other hand, it is well known that the thin lens equation is a simple, yet powerful tool in geometric optics design. An effort to expand the thin lens equation to cover a BSMI Gaussian beam was successful.[9] The resultant thin lens equation was still simple and effective, and has been accepted by optical industry.[10] However, this thin lens equation[9] considered only the case of a BSMI Gaussian beam. The effect of lens aperture truncation on the beam was not considered.

It has recently been shown that laser beams deviated somewhat from a BSMI Gaussian beam. The deviation can be described by the  $M^2$  (read  $M$  square) factor. A BSMI Gaussian beam has  $M^2 = 1$ , while a real laser beam has  $M^2 > 1$ . Measurement results showed that ion laser beams usually have an  $M^2$  factor in the range of 1.1 to 1.3. Collimated single-spatial-mode laser diode beams usually

have an  $M^2$  factor in the range $^{13}$  of 1.1 to 1.7. In both cases, the  $M^2$  factor increased when the truncation intensity level was increased. The  $M^2$  factor varies among different lasers and significantly affects the characteristics of a laser beam. Therefore the  $M^2$  factor cannot be easily neglected in optical design involving laser beams.

In this paper, we expand the work of Ref. 9 by deriving a new thin lens equation applicable to a real laser beam, including both the  $M^2$  factor and the effects of weak lens aperture truncation.

# 2 Gaussian Optics

It is helpful to review the basic characteristics of a BSMI Gaussian beam before starting our study. A BSMI Gaussian beam is described by

$$
w _ {G} (z) = w _ {0 G} \left[ 1 + \left(\frac {z \lambda}{\pi w _ {0 G} ^ {2}}\right) ^ {2} \right] ^ {1 / 2}, \tag {1}
$$

$$
R _ {G} (z) = z \left[ 1 + \left(\frac {\pi w _ {0 G} ^ {2}}{z \lambda}\right) ^ {2} \right], \tag {2}
$$

where  $w_{0G}$  is the  $1 / e^2$  intensity radius of the beam waist,  $w_{G}(z)$  is the  $1 / e^2$  intensity radius of the beam at an axial distance  $z$  from the beam waist,  $\lambda$  is the wavelength, and  $R_{G}(z)$  is the wavefront radius of the beam at  $z$ . The  $1 / e^2$  intensity far-field half-divergent angle  $\theta_{G}$  of this beam can be obtained by introducing a  $z \gg \pi w_{0G}^2 /\lambda$  to Eq. (1). The result is

$$
\theta_ {G} = \frac {w _ {G} (z)}{z} = \frac {\lambda}{\pi w _ {0 G}}. \tag {3}
$$

From Eq. (3) we have

$$
w _ {0 G} \theta_ {G} = \frac {\lambda}{\pi}. \tag {4}
$$

By the nature of a light wave,  $\lambda / \pi$  is the smallest possible value of the radius-divergence product. For a real laser beam, we have

$$
w _ {0} \theta = M ^ {2} \frac {\lambda}{\pi} > \frac {\lambda}{\pi}, \tag {5}
$$

where  $w_{0}$  and  $\theta$  are the  $1/e^{2}$  intensity waist radius and the far-field half-divergent angle of the real laser beam, respectively, and  $M^{2} > 1$ . Here  $M^{2}$  is considered a numerical description of the beam quality. A real laser beam can therefore be described by introducing an  $M^{2}$  factor into Eqs. (1) and (2):

$$
w (z) = w _ {0} \left[ 1 + \left(\frac {z \lambda M ^ {2}}{\pi w _ {0} ^ {2}}\right) ^ {2} \right] ^ {1 / 2}, \tag {6}
$$

$$
R (z) = z \left[ 1 + \left(\frac {\pi w _ {0} ^ {2}}{z \lambda M ^ {2}}\right) ^ {2} \right], \tag {7}
$$

where  $w(z)$  and  $R(z)$  are the  $1/e^2$  intensity radii of the beam and the beam wavefront at  $z$ , respectively. The farfield half-divergent angle  $\theta$  of this real laser beam can be obtained by introducing a  $z \gg \pi w_0^2/\lambda$  to Eq. (6). The result is

$$
\theta = \frac {w (z)}{z} = \frac {\lambda M ^ {2}}{\pi w _ {0}} \sim M ^ {2}. \tag {8}
$$

Equation (8) obviously agrees with Eq. (5). A real laser beam described by Eqs. (6) and (7) can be considered to contain a fictitious embedded BSMI Gaussian beam<sup>11</sup> described by

$$
w _ {e} (z) = \left(\frac {w _ {0}}{M}\right) \left\{1 + \left[ \frac {z \lambda}{\pi \left(w _ {0} / M\right) ^ {2}} \right] ^ {2} \right\} ^ {1 / 2}, \tag {9}
$$

$$
R _ {e} (z) = z \left\{1 + \left[ \frac {\pi \left(w _ {0} / M\right) ^ {2}}{z \lambda} \right] ^ {2} \right\}, \tag {10}
$$

where  $w_0 / M$ ,  $w_e(z)$ , and  $R_e(z)$  are the  $1/e^2$  intensity beam waist radius, the  $1/e^2$  intensity beam radius at  $z$ , and the wavefront radius at  $z$ , respectively. The far-field half-divergent angle  $\theta_e$  of the embedded BSMI Gaussian beam is given by introducing a  $z \gg \pi (w_0 / M)^2/\lambda$  to Eq. (9). The result is

$$
\theta_ {e} = \frac {w _ {e} (z)}{z} = \frac {\lambda}{\pi \left(w _ {0} / M\right)} \sim M. \tag {11}
$$

In this paper, we use the commonly used definition for the Rayleigh range  $z_{R}$  to describe the real laser beam, we have

$$
z _ {R} = \frac {\pi w _ {0} ^ {2}}{\lambda}. \tag {12}
$$

Equations (6), (7), and (12) form a complete set. In the following, we use Eqs. (6), (7), and (12) to denote a real

![](images/42159ffb5fb0270551d4215a9300e055daf93c3beb5db150a6130118587ca654.jpg)  
Fig. 1 A 2-D picture for laser beam input to and output from a finite aperture thin lens. The solid curves mark the  $1/e^2$  intensity contour. The lens truncates the beam at an intensity level lower than  $1/e^2$ .

laser beam input to a thin lens. Similarly, the real laser beam output from a thin lens can be described by Eqs. (13) to (15):

$$
w ^ {\prime} \left(z ^ {\prime}\right) = w _ {0} ^ {\prime} \left[ 1 + \left(\frac {z ^ {\prime} \lambda M ^ {\prime 2}}{\pi w _ {0} ^ {\prime 2}}\right) ^ {2} \right] ^ {1 / 2}, \tag {13}
$$

$$
R ^ {\prime} \left(z ^ {\prime}\right) = z ^ {\prime} \left[ 1 + \left(\frac {\pi w _ {0} ^ {\prime 2}}{z ^ {\prime} \lambda M ^ {\prime 2}}\right) ^ {2} \right], \tag {14}
$$

$$
z _ {R} ^ {\prime} = \frac {\pi w _ {0} ^ {\prime 2}}{\lambda}, \tag {15}
$$

where the prime symbol denotes the beam output from the thin lens. For  $M^2 = M'^2 = 1$ , Eqs. (6) and (7) and (13) and (14) reduce to the equations describing a BSMI Gaussian beam.

# 3 Derivation of the Thin Lens Equation

Figure 1 shows the situation for a real laser beam input to and output from a thin lens, where  $a$  is the radius of the lens aperture;  $o$  and  $i$  are the distances between the lens and the two beam waist, respectively;  $w(o)$  and  $w'(i)$  are the  $1/e^2$  intensity radii of the two beams at the lens, respectively; and  $(x,y,z)$  and  $(x',y',z')$  are the two coordinates originating at the centers of the two beam waist, respectively. The solid curves in Fig. 1 mark the  $1/e^2$  intensity contour of the beams, and the lens aperture truncates the input beam at the intensity level marked by the dash curves. In this paper, we consider only the situation where  $a/w(o) > 1$ . We call this "weak truncation." A laser beam may have different characteristics in the  $x$  and  $y$  directions (or  $x'$  and  $y'$  directions). We can separately study the beam characteristics in these two orthogonal directions. Thus in Fig. 1, we plot only a 2-D picture.

The effects of the lens and the lens aperture on the beam can be considered separately. First, the lens aperture weakly truncates the beam and changes the beam divergence from  $\theta$  to  $\theta^{\prime}$ , as shown in Fig. 2. Then, the lens focuses the truncated beam. Reference 7 considered a BSMI Gaussian beam and showed that  $\theta^{\prime}$  can be either larger or smaller than  $\theta$ , depending on the situation. Since Ref. 7 did not consider the effect of the  $M^2$  factor, it ex

![](images/0a7927cee4c3bafaa22652c0b89d8737fc3f9cf4ce54436c1dc988dbd974e28c.jpg)  
Fig. 2 Weak aperture truncation on a beam can change the far-field divergence of a Gaussian beam. The truncated beam can be approximated by another Gaussian beam. The solid curves are the  $1 / e^2$  intensity contour for a beam without aperture truncation. The dashed curves are the  $1 / e^2$  intensity contour for the same beam after being weakly truncated by an aperture.

plained the change of  $\theta$  by introducing a fictitious change in the beam waist. For a real laser beam, the situation of  $\theta^{\prime} > \theta$  can be explained by introducing an increment to either the beam waist or the  $M^2$  factor or both. The situation of  $\theta^{\prime} < \theta$  can be explained by introducing a decrement mainly to the beam waist because there is a lower limit of 1 to the  $M^2$  factor. We also expect that aperture truncation can reduce only the beam quality or increase the  $M^2$  factor. The

calculation of the changes in the beam waist and/or  $M^2$  factor caused by aperture truncation is beyond the scope of the thin lens equation. In this paper, we consider only the situation where  $\theta' > \theta$  and an increase in the  $M^2$  factor.

The thin lens approximations state that

$$
w (o) = w ^ {\prime} (i), \tag {16}
$$

$$
\frac {1}{R (o)} + \frac {1}{R ^ {\prime} (i)} = \frac {1}{f}, \tag {17}
$$

where  $f$  is the focal length of the lens used. Combining Eqs. (6), (7), and (12)-(17), we have derived a thin lens equation for a real laser beam as

$$
\frac {1}{(i / f)} + \frac {1}{(o / f) + (z _ {R} / M ^ {2} f) ^ {2} / [ (o / f) - 1 ]}
$$

$$
\begin{array}{l} - \frac {\left[ \left(M ^ {\prime 2} / M ^ {2}\right) ^ {2} - 1 \right] \left(z _ {R} / M ^ {2} f\right) ^ {2}}{\left[ \left(o / f\right) ^ {2} + \left(z _ {R} / M ^ {2} f\right) ^ {2} \right] \left[ \left(o / f\right) ^ {2} - \left(o / f\right) + \left(z _ {R} / M ^ {2} f\right) ^ {2} \right]} \\ = 1. \tag {18} \\ \end{array}
$$

The lens magnification ratio for a real laser beam has also been obtained from Eqs. (6), (7), and (12)-(17) as

$$
\frac {w _ {0} {} ^ {\prime}}{w _ {0}} = \frac {M ^ {\prime 2} / M ^ {2}}{\left[ (o / f) - 1 \right] ^ {2} + \left(z _ {R} / M ^ {2} f\right) ^ {2} + \left\{\left[\left(M ^ {\prime 2} / M ^ {2}\right) ^ {2} - 1 \right]\left(z _ {R} / M ^ {2} f\right) ^ {2} \right\} / \left[\left(o / f\right) ^ {2} + \left(z _ {R} / M ^ {2} f\right) ^ {2} \right]\left. \right) ^ {1 / 2}}. \tag {19}
$$

Equations (18) and (19) contain both  $M^2$  and  $M'^2$ . The derivations of Eqs. (18) and (19) are lengthy. The derivation details are shown in the Appendix of this paper. For  $M'^2 = M^2 > 1$ , Eqs. (18) and (19) reduce to, respectively,

$$
\frac {1}{(i / f)} + \frac {1}{(o / f) + \left(z _ {R} / M ^ {2} f\right) ^ {2} / [ (o / f) - 1 ]} = 1, \tag {20}
$$

and

$$
\frac {w _ {0} ^ {\prime}}{w _ {0}} = \frac {1}{\left\{\left[ (o / f) - 1 \right] ^ {2} + \left(z _ {R} / M ^ {2} f\right) ^ {2} \right\} ^ {1 / 2}}. \tag {21}
$$

For  $M'^2 = M^2 = 1$ , Eqs. (18) and (19) reduce to, respectively,

$$
\frac {1}{(i / f)} + \frac {1}{(o / f) + (z _ {R} / f) ^ {2} / [ (o / f) - 1 ]} = 1 \tag {22}
$$

and

$$
\frac {w _ {0} ^ {\prime}}{w _ {0}} = \frac {1}{\left\{\left[ \left(o / f\right) - 1 \right] ^ {2} + \left(z _ {R} / f\right) ^ {2} \right\} ^ {1 / 2}}. \tag {23}
$$

Equations (22) and (23) are identical to the equations previously obtained for a BSMI Gaussian beam. For  $z_{R} / M^{2} \to 0$ , the real laser beam approaches the geometric optics limit, and Eqs. (18) and (19) reduce to the geometric optics forms, respectively,

$$
\frac {1}{(i / f)} + \frac {1}{(o / f)} = 1, \tag {24}
$$

and

$$
\frac {w _ {0} ^ {\prime}}{w _ {0}} = \frac {1}{[ (o / f) - 1 ]}. \tag {25}
$$

It is interesting to see that  $M^2$  plays a role in reducing Eqs. (18) and (19) to Eqs. (24) and (25), while Eqs. (22) and (23) approach the geometric optics limit only when  $z_R \to 0$ . Equations (18)-(21) are new.

# 4 Results

In Figs. 3-5, we plot the  $i / f$  to  $o / f$  curves and the  $w_0^{\prime} / w_0$  to  $o / f$  curves obtained from Eqs. (18) and (19). Since laser users are often concerned about the beam focused spot size  $w_0^\prime$  as a function of the focusing distance  $i$  with  $z_{R}$  being a

![](images/eb5914952abebcb3e2d99d1357f780ecae92cc22306a3a52924e6e609efa0831.jpg)  
(a)

![](images/e6ca9da9afba5740cd523093dccccdeb45d4ba7e47ac738147f004849dc8b9f1.jpg)  
(b)

![](images/be1975c74e1f7b2fb59cf3ed0dd4dc62c5a39db46067dadac2e71c44b69eb7c7.jpg)  
(c)  
Fig. 3 Plots of  $i / f$  to  $o / f$  curves,  $w_0' / w_0$  to  $o / f$  curves, and  $w_0' / w_0$  to  $i / f$  curves for  $z_R / f = 0.01$  obtained from Eqs. (18) and (19). The solid curves are for  $M^2 = M'^2 = 1$  or  $M^2 = M'^2 = 1.5$ . The broken curves are for  $M^2 = 1$  and  $M'^2 = 1.2$  or  $M^2 = 1.5$  and  $M'^2 = 1.7$ . The broken lines are plotted to help show the symmetry of the curves. In (b) the solid curves for  $M^2 = M'^2 = 1$  and  $M^2 = M'^2 = 1.5$  have no discernible differences from the broken curves for  $M^2 = 1$  and  $M'^2 = 1.2$  and  $M^2 = 1.5$  and  $M'^2 = 1.7$ , respectively.

parameter, we also plot in Figs. 3-5 the  $w_0' / w_0$  to  $i / f$  curves with  $z_R / f$  as a parameter. Unfortunately we are unable to derive from Eqs. (6), (7), and (12)-(17) an analytical form of  $w_0' / w_0$  as a function of  $i / f$  and  $z_R / f$  even with  $M'^2 = M^2 = 1$ . We have therefore written a short computer program to solve Eqs. (18) and (19) for the  $w_0' / w_0$  to  $i / f$  curves. The value of  $z_R / f$  varies greatly in practical applications. For a single spatial mode laser diode  $w_0 \sim 1 \mu \mathrm{m}$  and  $\lambda \sim 1 \mu \mathrm{m}$ , and for most commonly used laser diode beam collimating lens  $f \sim 1 \mathrm{~mm}$ , we have  $z_R / f \sim 0.01$ . For an He-Ne laser beam of  $1 \mathrm{~mm}$  radius and a focusing lens of  $f = 50 \mathrm{~mm}$ , we have  $z_R / f \sim 100$ . In this paper, we consider three typical values of  $z_R / f = 0.01, 1,$  and 100, respectively.

The curves of  $i / f\sim o / f$ $w_0^{\prime} / w_0\sim o / f$  , and  $w_0^\prime /w\sim i / f$  with  $z_{R} / f = 0.01$  are plotted in Figs. 3(a)-3(b), respectively, for  $M^2 = M'^2 = 1$  (solid curve),  $M^2 = 1$  and  $M^{\prime 2}$ $= 1.2$  (broken curve),  $M^2 = M'^2 = 1.5$  (solid curve), and

$M^2 = 1.5$  and  $M^{\prime 2} = 1.7$  (broken curve). It can be seen from Fig. 3(a) that the  $i / f$  varying range for  $M^2 = M^{\prime 2}$ $= 1.5$  is larger than for  $M^2 = M^{\prime 2} = 1$ , and the  $i / f$  varying range decreases as  $M^{\prime 2}$  increases. Numerical calculation data shows that the two solid curves are symmetric about the point  $o / f = 1$  and  $i / f = 1$ , and the two broken curves are not symmetric about that point. The nonsymmetry of the two broken curves is not apparent because of the limited resolution of Fig. 3(a). It can be seen from Fig. 3(b) that the two  $w_0^{\prime} / w_0$  to  $o / f$  curves for  $M^2 = M^{\prime 2} = 1$  and  $M^2 = 1$  and  $M^{\prime 2} = 1.2$  have no discernible difference within the resolution of Fig. 3(b), as do the two  $w_0^{\prime} / w_0$  to  $o / f$  curves for  $M^2 = M^{\prime 2} = 1.5$  and  $M^2 = 1.5$  and  $M^{\prime 2} = 1.7$ . The  $w_0^{\prime} / w_0$  to  $o / f$  curve for  $M^2 = M^{\prime 2} = 1.5$  has a larger amplitude around line  $z / f = 1$  than the  $w_0^{\prime} / w_0$  to  $o / f$  curve for  $M^2 = M^{\prime 2}$ $= 1$ . Numerical calculation data shows that the  $w_0^{\prime} / w_0$  to  $o / f$  curves for  $M^2 = M^{\prime 2} = 1$  and  $M^2 = M^{\prime 2} = 1.5$  are sym

![](images/039d7b676fe879e76af7a489b4aa3445947e210bcf71c2602364e41ea50d71fb.jpg)  
(a)  
Fig. 4 Plots of  $i / f$  to  $o / f$  curves,  $w_0' / w_0$  to  $o / f$  curves, and  $w_0' / w_0$  to  $i / f$  curves for  $z_R / f = 1$  obtained from Eqs. (18) and (19). The solid curves are for  $M^2 = M'^2 = 1$  or  $M^2 = M'^2 = 1.5$ . The broken curves are for  $M^2 = 1$  and  $M'^2 = 1.2$  or  $M^2 = 1.5$  and  $M'^2 = 1.7$ . The broken lines are plotted to help show the symmetry of the curves.

![](images/56556b5771ac6034c0fa6a0f866da4eadd6fd2c115d664dfb14632afbdfb83a8.jpg)  
(b)

![](images/ba3f97ab1ba49f99acb7af0cc3a6e74e334097935c7eeab33198e9975b630f48.jpg)  
(c)

![](images/4474d30a367eb72b28c394e75a3e4d01aeb9dc42b872d24c0d6981900fe8d543.jpg)  
(a)

![](images/56b5bfe266f197e6cbf8c3a8ec0081b91dfb8d7f0c113fddd7ca4e064cda26a8.jpg)  
(b)

![](images/82600d66a721facc2063015a924ba143da22f1422ddc85e8199cd23788dc1ab3.jpg)  
(c)  
Fig. 5 Plots of  $i / f$  to  $o / f$  curves,  $w_0' / w_0$  to  $o / f$  curves, and  $w_0' / w_0$  to  $i / f$  curves for  $z_R / f = 100$  obtained from Eqs. (18) and (19). The solid curves are for  $M^2 = M'^2 = 1$  or  $M^2 = M'^2 = 1.5$ . The broken curves are for  $M^2 = 1$  and  $M'^2 = 1.2$  or  $M^2 = 1.5$  and  $M'^2 = 1.7$ . The broken lines are plotted to help show the symmetry of the curves. In (a) the solid curves for  $M^2 = M'^2 = 1$  and  $M^2 = M'^2 = 1.5$  have no discernible differences from the broken curves for  $M^2 = 1$  and  $M'^2 = 1.2$  and  $M^2 = 1.5$  and  $M'^2 = 1.7$ , respectively.

metric about the line  $z / f = 1$ , while the  $w_0' / w_0$  to  $o / f$  curves for  $M^2 = 1$  and  $M'^2 = 1.2$  and  $M^2 = 1.5$  and  $M'^2 = 1.7$  are not symmetric about the line  $o / f = 1$ . But the nonsymmetry is not apparent because of the limited resolution of Fig. 3(b). The smaller varying range of  $i / f$  for the broken curves can also be seen in Fig. 3(c). Numerical calculation data show that the solid curves are symmetric about line  $i / f = 1$ , but the broken curves are not.

The curves of  $i / f$  to  $o / f$ ,  $w_0' / w_0$  to  $o / f$ , and  $w_0' / w$  to  $i / f$  with  $z_R / f = 1$  are plotted in Figs. 4(a) to 4(c), respectively, for  $M^2 = M'^2 = 1$  (solid curve),  $M^2 = 1$  and  $M'^2 = 1.2$  (broken curve),  $M^2 = M'^2 = 1.5$  (solid curve), and  $M^2 = 1.5$  and  $M'^2 = 1.7$  (broken curve). These curves have characteristics similar to the curves shown in Fig. 3, except that the nonsymmetry of all the broken curves shown in Fig. 4 are more apparent.

The curves of  $i / f$  to  $o / f$ ,  $w_0' / w_0$  to  $o / f$ , and  $w_0' / w$  to  $i / f$  with  $z_R / f = 100$  are plotted in Figs. 5(a) to 5(c), respectively, for  $M^2 = M'^2 = 1$  (solid curve),  $M^2 = 1$  and  $M'^2 = 1.2$  (broken curve),  $M^2 = M'^2 = 1.5$  (solid curve), and  $M^2 = 1.5$  and  $M'^2 = 1.7$  (broken curve). These curves have characteristics similar to the curves shown in Figs. 3 and 4, except that in Fig. 5(a) the solid curves and the broken curves are not discernible within the resolution of the figure, and in Figs. 5(b) and (c) the nonsymmetry of the broken curves are not apparent.

Figures 3-5 show that the value of the  $M^2$  factor and the change of the  $M^2$  factor have significant effects on the beam characteristics and should be included in the thin lens equation.

Three important results can be found from Figs. 3-5.

1. For any given values of  $o / f$ , the values of the corresponding  $i / f$  shown by the broken curves for

$M^{\prime 2} > M^{2}$  are smaller than those shown by the solid curves for  $M^{\prime 2} = M^{2}$ . This result indicates that with lens aperture truncation on the beam, the focal point is shifted toward the lens.

2. The focused spot sizes  $w_0' / w_0$  shown by the broken curves for  $M'^2 > M^2$  are larger than those shown by the solid curves for  $M'^2 = M^2$ . This result indicates that with lens aperture truncation on the beam the focused spot size is increased.

3. The  $i / f$  to  $o / f$  curves for  $M^{\prime 2} > M^{2}$  are not symmetric about line  $z / f = 1$ . This result indicates that with lens aperture truncation on the beam, the beam has a nonsymmetric characteristics in the focusing space.

Results 1 to 3 are similar to the results previously derived from the diffraction optics theory.

# 5 Two Application Examples

# 5.1 Measurement of the Far-Field Divergence of a Real Laser Beam

In this section, we apply the new thin lens equations to the measurement of the far-field divergence of a real laser beam.

A schematic of a commonly used method for measuring beam far-field divergence angle is shown in Fig. 6, where a lens is used to focus the beam to be measured, a beam profiler is positioned at the lens focal point to measure the beam size  $w^{\prime}(i - f)$ , and the beam far-field divergence angle  $\theta$  can be calculated as

$$
\theta = \frac {w ^ {\prime} (i - f)}{f}. \tag {26}
$$

![](images/b9b9df84a1225629e9803e9a7812a0de9a7cce7d3e33102b6127f8a65d7d32c6.jpg)  
Fig. 6 Schematic of a setup for measuring laser beam far-field divergence angle  $\theta$ .

Equation (26) is widely used and has been shown to be applicable to a BSMI Gaussian beam. In such a measurement, the lens is chosen so that the lens aperture does not truncate the beam, therefore we have  $M'^2 = M^2 > 1$ , and Eqs. (20) and (21) are applicable. Inserting Eqs. (13) and (21) into Eq. (26) to eliminate  $w'(i - f)$  and  $w_0'$ , we obtain

$$
\begin{array}{l} \theta = \frac {w ^ {\prime} (i - f)}{f} \\ = \frac {w _ {0} ^ {2}}{\left[ (o / f) - 1 \right] ^ {2} + \left(z _ {R} / M ^ {2} f\right) ^ {2}} \\ + \frac {(i - f) ^ {2} \lambda^ {2} M ^ {4}}{\pi^ {2}} \frac {\left[ (o / f) - 1 \right] ^ {2} + \left(z _ {R} / M ^ {2} f\right) ^ {2}}{w _ {0} ^ {2}}. \tag {27} \\ \end{array}
$$

Solving Eq. (20) for  $i - f$ , we obtain

$$
i - f = \frac {s - f}{[ (o / f) - 1 ] ^ {2} + \left(z _ {R} / M ^ {2} f\right) ^ {2}}. \tag {28}
$$

Inserting Eq. (28) into Eq. (27), and rearranging the result, we obtain

$$
\theta = \frac {w ^ {\prime} (i - f)}{f} = \frac {\lambda M ^ {2}}{\pi w _ {0}}. \tag {29}
$$

Equation (29) is identical to Eq. (8). Thus we prove that Eqs. (20) and (21) are consistent and correct when being applied to the measurement of the far-field divergence of a real laser beam described by Eqs. (6), (7), and (12).

# 5.2 Focal Shift

If the lens aperture truncates the beam, the situation is more complex. There are no direct comparisons between the results obtained in this paper and results obtained previously. In this section, we try to compare indirectly one result obtained in this paper with the results obtained previously.

Reference 7 showed that after being weakly truncated, a BSMI Gaussian beam with a  $1/e$  far-field half-divergent angle  $\theta_{m}$  could be approximated by another BSMI Gaussian beam with a  $1/e$  intensity far-field divergence angle of  $\theta_{m}^{\prime}$ . Figure 5(a) of Ref. 7 plotted  $\theta_{m}/\theta_{m}^{\prime}$  against  $a/r_{0}$  with  $p=l\lambda/r_{m}^{2}$  as a parameter, where  $a$  is the aperture radius,  $r_{0}$

![](images/a8013516b8378de8331c69b945063a0e06e95667da5e5921a29e02d5702ef8c0.jpg)  
Fig. 7 Focal shift caused by lens aperture truncation. The dashed curve is for  $Z / Z_{0}$  against  $a / w(o)$  obtained previously using diffraction optics theory. The solid curve is for  $i_{M^{\prime}2 > M^{2}} / i_{M^{\prime}2 = M^{2}}$  against  $a / w(o)$  obtained using the new thin lens equation.

is the beam  $1 / e$  radius at the aperture,  $l$  is the axial distance between the beam waist and the aperture, and  $r_m$  is the beam waist  $1 / e$  intensity radius. The symbols  $\theta_{m}$  and  $\theta_{m}^{\prime}$  can be related to symbols  $\theta_{e}$  and  $\theta_{e}^{\prime}$  used in this paper because (1)  $\theta_{e}$  and  $\theta_{e}^{\prime}$  also describe BSMI Gaussian beams and (2) the embedded Gaussian beam is a good approximation of the real laser beam studied in this paper. Considering that  $\theta_{m}$  and  $\theta_{m}^{\prime}$  were defined at the  $1 / e$  intensity level, we have that  $\theta_{m} = \theta_{e} / \sqrt{2},\theta_{m}^{\prime} = \theta_{e}^{\prime} / \sqrt{2}$ , and

$$
\frac {\theta_ {m}}{\theta_ {m} ^ {\prime}} = \frac {\theta_ {e}}{\theta_ {e} ^ {\prime}} = \frac {M}{M ^ {\prime}}. \tag {30}
$$

The other symbols used in Ref. 7 can be related to the symbols used in this paper by the conversions  $r_m = w_0 / \sqrt{2}$ ,  $r_0 = w(o) / \sqrt{2}$ , and  $p = l\lambda / r_m^2 = 2o\lambda / w_0^2 = 2\pi o / z_R$ . To quickly estimate the value of  $p$  for common applications, we consider  $o \sim 10 \mathrm{~mm}$ ,  $\lambda \sim 0.001 \mathrm{~mm}$ , and  $w_0 \sim 1 \mathrm{~mm}$ , and obtain  $p \sim 0.01$ . Then we can find  $M / M' = \theta_m / \theta_m'$  as a function of  $a / w(o) = a / \sqrt{2} r_0$  from the  $p = 0,1$  curves in Fig. 5(a) of Ref. 7. For example, we have  $M / M' \approx 0.75$  at the lower limit  $a / r_0 \approx 1.6$  of Fig. 5(a) of Ref. 7, which is equivalent to  $a / w(o) \approx 1.1$  in this paper. At  $a / r_0 > 3.5$  in Fig. 5(a) of Ref. 7, which is equivalent to  $a / w(o) > 2.5$ , we have  $M / M' \approx 1$ .

Reference 3 studied the focusing of a BSMI Gaussian beam through a finite aperture lens. Figure 1 of Ref. 3 plotted the focal shift  $Z / Z_{0}$  as a function of  $a / w_{1}$  with  $P$  and  $\xi_0$  as parameters. To relate the symbols used in Ref. 3 to the symbols used in this paper, we have  $Z_{0} = i_{M^{\prime}2 = M^{2}}$ , where  $i_{M^{\prime}2 = M^{2}}$  denotes  $i$  for a beam without lens aperture truncation;  $Z = i_{M^{\prime}2 > M^{2}}$ , where  $i_{M^{\prime}2 > M^{2}}$  denotes  $i$  for a beam with lens aperture truncation,  $Z / Z_{0} = i_{M^{\prime}2 > M^{2}} / i_{M^{\prime}2 = M^{2}}$ ;  $a$  used in Ref. 3 is identical to the  $a$  used in this paper;  $w_{1} = w(o)$ ;  $a / w_{1} = a / w(o)$ ;  $\xi_0 = o / z_R$ ;  $w_{s} = w_{0}$ , and  $P = kw_{s}^{2} / f = (2\pi /\lambda)w_{0}^{2} / f = 2z_{R} / f$ . Now we are ready to compare the results. The comparison procedure is

1. Read off the focal shift  $Z / Z_{0}$  as a function of lens aperture truncation level  $a / w_{1}$  from Fig. 1 of Ref. 3.  
2. Convert the  $a / w_{1}$  obtained in step 1 to  $a / r_0$  used in Ref. 7.  
3. Read off the  $M / M^{\prime} = \theta_{m} / \theta_{m}^{\prime}$  as a function of  $a / r_0$  from Fig. 5(a) of Ref. 7.  
4. Assume  $M^2 = 1$ , and input the  $M'^2$  obtained in step 3 to Eq. (18) of this paper to calculate the focal shift  $i_{M'^2 > M^2} / i_{M'^2 = M^2}$ .  
5. Compare  $Z / Z_{0}$  obtained in step 1 with  $i_{M^{\prime}2 > M^{2}} / i_{M^{\prime}2 = M^{2}}$  obtained in step 4.

The results are plotted in Fig. 7. It is interesting to see that the difference between  $Z / Z_{0}$  and  $i_{M^{\prime}2 > M^{2}} / i_{M^{\prime}2 = M^{2}}$  is so small even though they are obtained in a completely different way.

# 6 Conclusion

In this paper, we derived a thin lens equation for a real laser beam with  $M^2 > 1$ . The effect of week lens aperture truncation on the beam is considered by introducing an increment to the  $M^2$  factor. The new thin lens equation is more accurate than the thin lens equation obtained previously and still simple, and can be readily applied to a real laser beam without truncation. The thin lens equation can be applied to a truncated real laser beam when diffraction optics can provide information about the changes in the beam waist and the  $M^2$  factor caused by truncation. In this case, the new thin lens equation can be used to estimate the phenomena of focal shift and focused spot size change caused by weak lens aperture truncation. Previously, these phenomena could be described only by diffraction optics theory.

The new thin lens equation is only an approximation. For example, lens aperture truncation on a beam can generate diffraction patterns that vary in a complex manner and are more than what a thin lens equation can handle. Neither can the new thin lens equation predict the beam divergence change caused by truncation. Therefore, the new thin lens equation cannot be used to replace the numerical analysis of diffraction optics, particularly in the situation where the lens aperture strongly truncates the beam.

# 7 Appendix

# 7.1 Derivation of Eq. (18)

Combining Eqs. (6), (13), and (16), we obtain

$$
w _ {0} \left[ 1 + \left(\frac {o M ^ {2}}{z _ {R}}\right) ^ {2} \right] ^ {1 / 2} = w _ {0} ^ {\prime} \left[ 1 + \left(\frac {i M ^ {\prime 2}}{z _ {R} ^ {\prime}}\right) ^ {2} \right] ^ {1 / 2}. \tag {31}
$$

Combining Eqs. (12), (15), and (31) we obtain

$$
z _ {R} \left[ 1 + \left(\frac {o M ^ {2}}{z ^ {R}}\right) ^ {2} \right] = z _ {R} ^ {\prime} \left[ 1 + \left(\frac {i M ^ {\prime 2}}{z _ {R} ^ {\prime}}\right) ^ {2} \right]. \tag {32}
$$

Reorganizing Eq. (32), we obtain

$$
\left(\frac {o M ^ {4}}{z _ {R}}\right) o \left[ 1 + \left(\frac {z _ {R}}{o M ^ {2}}\right) ^ {2} \right] = \left(\frac {i M ^ {\prime 4}}{z _ {R} ^ {\prime}}\right) i \left[ 1 + \left(\frac {z _ {R} ^ {\prime}}{i M ^ {\prime 2}}\right) ^ {2} \right]. \tag {33}
$$

Inserting Eqs. (7) and (14), respectively, into the left and right sides of Eq. (33) we obtain

$$
\left(\frac {o M ^ {4}}{z _ {R}}\right) R (o) = \left(\frac {i M ^ {\prime 4}}{z _ {R} ^ {\prime}}\right) R ^ {\prime} (i). \tag {34}
$$

Reorganizing Eq. (34) we obtain

$$
i \left[ 1 + \left(\frac {z _ {R} ^ {\prime}}{i M ^ {\prime 2}}\right) ^ {2} \right] = i \left\{1 + \left[ \frac {M ^ {\prime 2} R ^ {\prime} (i) z _ {R}}{M ^ {4} o R (o)} \right] ^ {2} \right\}. \tag {35}
$$

The left side of Eq. (35) is  $R^{\prime}(i)$ , and we have

$$
R ^ {\prime} (i) = i \left\{1 + \left[ \frac {M ^ {\prime 2} R ^ {\prime} (i) z _ {R}}{M ^ {4} o R (o)} \right] ^ {2} \right\}. \tag {36}
$$

Using Eq. (17) to solve for  $R^{\prime}(i)$  and inserting the result into both sides of Eq. (36) we obtain

$$
\frac {f R (o)}{R (o) - f} = i + \frac {M ^ {\prime 4} i f ^ {2} z _ {R} ^ {2}}{M ^ {8} o ^ {2} [ R (o) - f ] ^ {2}}. \tag {37}
$$

Inserting Eqs. (7) into Eq. (37) to eliminate  $R(o)$  and reorganizing the result, we obtain

$$
\begin{array}{l} M ^ {4} o ^ {2} f + f z _ {R} ^ {2} - M ^ {4} o f ^ {2} - M ^ {4} o ^ {2} i - i z _ {R} ^ {2} + 2 M ^ {4} o i f - M ^ {4} i f ^ {2} \\ = \frac {\left(M ^ {4} - M ^ {4}\right) i f ^ {2} z _ {R} ^ {2}}{M ^ {4} o ^ {2} + z _ {R} ^ {2}}. \tag {38} \\ \end{array}
$$

Reorganizing the left side of Eq. (38) we obtain

$$
M ^ {4} (o - f) \left(o f - o i + i f\right) + z _ {R} ^ {2} (f - i) = \frac {\left(M ^ {4} - M ^ {4}\right) i f ^ {2} z _ {R} ^ {2}}{M ^ {4} o ^ {2} + z _ {R} ^ {2}}. \tag {39}
$$

Dividing both sides of Eq. (39) by term  $M^4 (o - f)if$  and reorganizing the result, we obtain

$$
\begin{array}{l} \left(\frac {1}{o}\right) + \left[ \left(\frac {1}{i}\right) - \left(\frac {1}{f}\right) \right] \left[ 1 + \frac {z _ {R} ^ {2}}{M ^ {4} o (o - f)} \right] \\ = \frac {\left[ \left(M ^ {\prime} / M\right) ^ {4} - 1 \right] f z _ {R} ^ {2}}{o (o - f) \left(M ^ {4} o ^ {2} + z _ {R} ^ {2}\right)}. \tag {40} \\ \end{array}
$$

Multiplying both sides of Eq. (40) by term  $f / [1 + z_{R}^{2} / M^{4}o(o - f)]$  and reorganizing the result, we obtain Eq. (18).

# 7.2 Derivation of Eq. (19)

Combining Eqs. (34) and (17) to eliminate  $R^{\prime}(i)$  and reorganizing the result, we obtain

$$
\frac {z _ {R}}{z _ {R} ^ {\prime}} = \frac {M ^ {4} o [ R (o) - f ]}{M ^ {\prime 4} f ^ {2}} \frac {1}{(i / f)}. \tag {41}
$$

Inserting Eqs. (7) and (18) into Eq. (41) to eliminate  $R(o)$  and  $1 / (i / f)$ , respectively, and then reorganizing the result we obtain

$$
\frac {z _ {R}}{z _ {R} ^ {\prime}} = \frac {\left[ (o / f) - 1 \right] ^ {2} + \left(z _ {R} / M ^ {2} f\right) ^ {2} + \left\{\left[ \left(M ^ {\prime} / M\right) ^ {4} - 1 \right] \left(z _ {R} / M ^ {2} f\right) ^ {2} \right\} / \left[ \left(o / f\right) ^ {2} + \left(z _ {R} / M ^ {2} f\right) ^ {2} \right]}{\left(M ^ {\prime} / M\right) ^ {4}}. \tag {42}
$$

From Eqs. (12) and (15) we know that  $w_0' / w_0 = (z_R' / z_R)^{1/2}$ , then we obtain Eq. (19) from Eq. (42).

# References

1. Y. Li, "Degeneracy and regeneracy in the axial field of a focused truncated Gaussian beam," J. Opt. Soc. Am. A 5, 1397-1406 (1988).  
2. Y. Li, "Oscillations and discontinuity in the focal shift of Gaussian laser beams," J. Opt. Soc. Am. A 3, 1761-1765 (1986).  
3. K. Tanaka, N. Saga, and K. Hauchi, “Focusing of a Gaussian beam through a finite aperture lens,” Appl. Opt. 24, 1098-1101 (1985).  
4. V. Mahajan, "Axial irradiance and optimum focusing of laser beam," Appl. Opt. 22, 3042-3053 (1983).  
5. N. Saga, K. Tanaka, and O. Fukumitsu, "Diffraction of a Gaussian beam through a finite aperture lens and the resulting heterodyne efficiency," Appl. Opt. 20, 2817-2831 (1981).  
6. Y. Li, "Degeneracy in the Fraunhofer diffraction of truncated Gaussian beams," J. Opt. Soc. Am. A 4, 1237-1242 (1987).  
7. P. Belland and J. Crenn, "Changes in the characteristics of a Gaussian beam weakly diffracted by a circular aperture," Appl. Opt. 21, 522-527 (1982).  
8. R. Herloski, S. Marshall, and R. Antos, "Gaussian beam ray-equivalent modeling and optical design," Appl. Opt. 22, 1168-1174 (1983).  
9. S. Self, “Focusing of spherical Gaussian beams,” Appl. Opt. 22, 658-661 (1983).  
10. "Gaussian beam optics," Melles Griot catalog, Irvine, CA (1995).  
11. A. Siegman, "New developments in laser resonators," Proc. SPIE 1224, 2-14 (1990).  
12. M. Sasnett and T. Johnston, Jr., "Beam characterization and measurement of propagation attributes," Proc. SPIE 1414, 21-32 (1991).  
13. H. Sun, Unpublished Report, Coherent Inc., Auburn, CA (1997).  
14. H. Sun, "On the measurement of beam far field divergence," submitted for publication.

![](images/73e04913cd411c9d53ac1edfdb6033f9f0a3343e743a9cedad9c6377f16cf621.jpg)

Haiyin Sun received his BSc in physics in 1982 from the Shanghai Teacher's University (STU), his MSc in laser and optics in 1985 from the Shanghai Institute of Optics and Fine Mechanics (SIOF), Chinese Academy of Science, and his PhD in lasers and optics in 1994 from the University of Arkansas at Little Rock (ULAR). In 1982 he was a physics instructor at STU, from 1986 to 1988 he was a research assistant professor of lasers and optics at SIOF,

from 1989 to 1990 he was a visiting scientist in the Optical Network Research Center of F. R. Germany's Post at Darmstadt, from 1994 to 1996 he was an optical engineer with Power Technology, Inc., in Mabelvale, Arkansas, and in August 1996 he joined Coherent, Inc., Auburn Group in Auburn, California, where he is currently the product manager of optics and laser diodes. He is also an adjunct assistant professor of applied science at UALR. He will be listed in the 1999 editions five Who's Who series books. Dr. Sun's main research interests are laser diode dynamics and modules, optical measurements and instrumentation, and mathematical modeling. In these fields, he is the primary author of over 20 referred journal papers and a book chapter.