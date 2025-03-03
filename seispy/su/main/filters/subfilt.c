/* Copyright (c) Colorado School of Mines, 2011.*/
/* All rights reserved.                       */

/* SUBFILT: $Revision: 1.22 $ ; $Date: 2012/11/28 22:13:13 $	*/

#include "su.h"
#include "segy.h"
#include "su_filters.h"

/*********************** self documentation **********************/
char *sdoc[] = {
" 								",
" SUBFILT - apply Butterworth bandpass filter 			",
" 								",
" subfilt <stdin >stdout [optional parameters]			",
" 							        ",
" Required parameters:						",
" 	if dt is not set in header, then dt is mandatory	",
" 							        ",
" Optional parameters: (nyquist calculated internally)		",
" 	zerophase=1		=0 for minimum phase filter 	",
" 	locut=1			=0 for no low cut filter 	",
" 	hicut=1			=0 for no high cut filter 	",
" 	fstoplo=0.10*(nyq)	freq(Hz) in low cut stop band	",
" 	astoplo=0.05		upper bound on amp at fstoplo 	",
" 	fpasslo=0.15*(nyq)	freq(Hz) in low cut pass band	",
" 	apasslo=0.95		lower bound on amp at fpasslo 	",
" 	fpasshi=0.40*(nyq)	freq(Hz) in high cut pass band	",
" 	apasshi=0.95		lower bound on amp at fpasshi 	",
" 	fstophi=0.55*(nyq)	freq(Hz) in high cut stop band	",
" 	astophi=0.05		upper bound on amp at fstophi 	",
" 	verbose=0		=1 for filter design info 	",
" 	dt = (from header)	time sampling interval (sec)	",
" 							        ",
" ... or  set filter by defining  poles and 3db cutoff frequencies",
"	npoleselo=calculated     number of poles of the lo pass band",
"	npolesehi=calculated     number of poles of the lo pass band",
"	f3dblo=calculated	frequency of 3db cutoff frequency",
"	f3dbhi=calculated	frequency of 3db cutoff frequency",
" 							        ",
" Notes:						        ",
" Butterworth filters were originally of interest because they  ",
" can be implemented in hardware form through the combination of",
" inductors, capacitors, and an amplifier. Such a filter can be ",
" constructed in such a way as to have very small oscillations	",
" in the flat portion of the bandpass---a desireable attribute.	",
" Because the filters are composed of LC circuits, the impulse  ",
" response is an ordinary differential equation, which translates",
" into a polynomial in the transform domain. The filter is expressed",
" as the division by this polynomial. Hence the poles of the filter",
" are of interest.					        ",
" 							        ",
" The user may define low pass, high pass, and band pass filters",
" that are either minimum phase or are zero phase.  The default	",
" is to let the program calculate the optimal number of poles in",
" low and high cut bands. 					",
" 							        ",
" Alternately the user may manually define the filter by the 3db",
" frequency and by the number of poles in the low and or high	",
" cut region. 							",
" 							        ",
" The advantage of using the alternate method is that the user  ",
" can control the smoothness of the filter. Greater smoothness  ",
" through a larger pole number results in a more bell shaped    ",
" amplitude spectrum.						",
" 							        ",
" For simple zero phase filtering with sin squared tapering use ",
" \"sufilter\".						        ",
NULL};

/* Credits:
 *	CWP: Dave Hale c. 1993 for bf.c subs and test drivers
 *	CWP: Jack K. Cohen for su wrapper c. 1993
 *      SEAM Project: Bruce Verwest 2009 added explicit pole option
 *                    in a program called "subfiltpole"
 *      CWP: John Stockwell (2012) combined Bruce Verwests changes
 *           into the original subfilt.
 *
 * Caveat: zerophase will not do good if trace has a spike near
 *	   the end.  One could make a try at getting the "effective"
 *	   length of the causal filter, but padding the traces seems
 *	   painful in an already expensive algorithm.
 *
 *
 * Theory:
 * The 
 *
 * Trace header fields accessed: ns, dt, trid
 */
/**************** end self doc ***********************************/

void bfhighpass_trace(int zerophase, int npoles, float f3db, spy_trace *tr_in, spy_trace *tr){

    if (tr == NULL){
        tr = tr_in;
    }
    unsigned short nt = tr->hdr.n_sample;
    bfhighpass(npoles,f3db,nt,tr_in->data,tr->data);
    if (zerophase) {
    register int i;
        for (i=0; i<nt/2; ++i) { /* reverse trace in place */
        register float tmp = tr->data[i];
        tr->data[i] = tr->data[nt-1 - i];
        tr->data[nt-1 - i] = tmp;
    }
        bfhighpass(npoles,f3db,nt,tr->data,tr->data);
        for (i=0; i<nt/2; ++i) { /* flip trace back */
        register float tmp = tr->data[i];
        tr->data[i] = tr->data[nt-1 - i];
        tr->data[nt-1 - i] = tmp;
    }
    }
}

void bflowpass_trace(int zerophase, int npoles, float f3db, spy_trace *tr_in, spy_trace *tr){
    if (tr == NULL){
        tr = tr_in;
    }
    unsigned short nt = tr->hdr.n_sample;
    bflowpass(npoles,f3db,nt,tr_in->data,tr->data);
    if (zerophase) {
        register int i;
        for (i=0; i<nt/2; ++i) { /* reverse trace */
            register float tmp = tr->data[i];
            tr->data[i] = tr->data[nt-1 - i];
            tr->data[nt-1 - i] = tmp;
        }
        bflowpass(npoles,f3db,nt,tr->data,tr->data);
            for (i=0; i<nt/2; ++i) { /* flip trace back */
            register float tmp = tr->data[i];
            tr->data[i] = tr->data[nt-1 - i];
            tr->data[nt-1 - i] = tmp;
        }
    }
}
