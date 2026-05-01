# mersenne2 mixed CRT test

Small experimental CPU code based on Yves Gallot mersenne2.
Original code is here:
https://github.com/galloty/mersenne2

This version try to test odd transform size like 9 * 2^m or 21 * 2^m.
The odd part is separated with CRT indexing.
The power of two part keep the half real GF(p^2) transform.

It use GF(M61^2) and GF(M31^2), then reconstruct with CRT/Garner.
Radix 3, 7, 9, 21 and 63 are experimental.
Odd roots are real scalar, so multiplication is cheaper.

Compile on Linux:

```bash
g++ -std=c++17 -O3 -march=native mersenne2_mixed_crt_2d_half_fast_scalarodd.cpp -o mersenne2_mixed_crt_2d_half_fast
```

Compile on Mac:

```bash
clang++ -std=c++17 -O3 -march=native mersenne2_mixed_crt_2d_half_fast_scalarodd.cpp -o mersenne2_mixed_crt_2d_half_fast
```

Run:

```bash
./mersenne2_mixed_crt_2d_half_fast 11213
./mersenne2_mixed_crt_2d_half_fast 11213 --original
./mersenne2_mixed_crt_2d_half_fast 11213 --compare
```

Example:

```text
M86243 use 9*2^8 instead of 2^12
storage is 1152 complex values instead of 2048
```

# Mixed radix size table

This table compare the normal power of two size with the experimental mixed CRT size.

The ranges use the same safe size formula used by the test program:


```text
log2(n) + 2*(q/n + 1) < 92
```

`mersenne2 original` means the normal power of two size. `mixed CRT` means the size selected by `mersenne2_mixed_crt_2d_half_fast.cpp`.


The last column is only a practical note from CPU tests. GPU can be different.


| q range | mersenne2 original | mixed CRT size | storage original | storage mixed | gain vs original | note |
|:---:|---:|---:|---:|---:|---:|:---|
| 9 206 - 10 334 | 2<sup>8</sup> | **63*2<sup>2</sup>** | 2<sup>7</sup> | 63*2<sup>1</sup> | x1.016 | --original probably faster |
| 10 335 - 10 496 | 2<sup>8</sup> | **2<sup>8</sup>** | 2<sup>7</sup> | 2<sup>7</sup> | x1.000 | --original / same size |
| 10 497 - 11 783 | 2<sup>9</sup> | **9*2<sup>5</sup>** | 2<sup>8</sup> | 9*2<sup>4</sup> | x1.778 | mixed probably faster |
| 11 784 - 13 710 | 2<sup>9</sup> | **21*2<sup>4</sup>** | 2<sup>8</sup> | 21*2<sup>3</sup> | x1.524 | mixed probably faster |
| 13 711 - 15 631 | 2<sup>9</sup> | **3*2<sup>7</sup>** | 2<sup>8</sup> | 3*2<sup>6</sup> | x1.333 | mixed probably faster |
| 15 632 - 18 187 | 2<sup>9</sup> | **7*2<sup>6</sup>** | 2<sup>8</sup> | 7*2<sup>5</sup> | x1.143 | --original probably faster |
| 18 188 - 20 417 | 2<sup>9</sup> | **63*2<sup>3</sup>** | 2<sup>8</sup> | 63*2<sup>2</sup> | x1.016 | --original probably faster |
| 20 418 - 20 736 | 2<sup>9</sup> | **2<sup>9</sup>** | 2<sup>8</sup> | 2<sup>8</sup> | x1.000 | --original / same size |
| 20 737 - 23 279 | 2<sup>10</sup> | **9*2<sup>6</sup>** | 2<sup>9</sup> | 9*2<sup>5</sup> | x1.778 | mixed probably faster |
| 23 280 - 27 084 | 2<sup>10</sup> | **21*2<sup>5</sup>** | 2<sup>9</sup> | 21*2<sup>4</sup> | x1.524 | mixed probably faster |
| 27 085 - 30 879 | 2<sup>10</sup> | **3*2<sup>8</sup>** | 2<sup>9</sup> | 3*2<sup>7</sup> | x1.333 | mixed probably faster |
| 30 880 - 35 926 | 2<sup>10</sup> | **7*2<sup>7</sup>** | 2<sup>9</sup> | 7*2<sup>6</sup> | x1.143 | --original probably faster |
| 35 927 - 40 331 | 2<sup>10</sup> | **63*2<sup>4</sup>** | 2<sup>9</sup> | 63*2<sup>3</sup> | x1.016 | --original probably faster |
| 40 332 - 40 960 | 2<sup>10</sup> | **2<sup>10</sup>** | 2<sup>9</sup> | 2<sup>9</sup> | x1.000 | --original / same size |
| 40 961 - 45 982 | 2<sup>11</sup> | **9*2<sup>7</sup>** | 2<sup>10</sup> | 9*2<sup>6</sup> | x1.778 | mixed probably faster |
| 45 983 - 53 496 | 2<sup>11</sup> | **21*2<sup>6</sup>** | 2<sup>10</sup> | 21*2<sup>5</sup> | x1.524 | mixed probably faster |
| 53 497 - 60 990 | 2<sup>11</sup> | **3*2<sup>9</sup>** | 2<sup>10</sup> | 3*2<sup>8</sup> | x1.333 | mixed probably faster |
| 60 991 - 70 956 | 2<sup>11</sup> | **7*2<sup>8</sup>** | 2<sup>10</sup> | 7*2<sup>7</sup> | x1.143 | --original probably faster |
| 70 957 - 79 654 | 2<sup>11</sup> | **63*2<sup>5</sup>** | 2<sup>10</sup> | 63*2<sup>4</sup> | x1.016 | --original probably faster |
| 79 655 - 80 896 | 2<sup>11</sup> | **2<sup>11</sup>** | 2<sup>10</sup> | 2<sup>10</sup> | x1.000 | --original / same size |
| 80 897 - 90 812 | 2<sup>12</sup> | **9*2<sup>8</sup>** | 2<sup>11</sup> | 9*2<sup>7</sup> | x1.778 | mixed probably faster |
| 90 813 - 105 648 | 2<sup>12</sup> | **21*2<sup>7</sup>** | 2<sup>11</sup> | 21*2<sup>6</sup> | x1.524 | mixed probably faster |
| 105 649 - 120 445 | 2<sup>12</sup> | **3*2<sup>10</sup>** | 2<sup>11</sup> | 3*2<sup>9</sup> | x1.333 | mixed probably faster |
| 120 446 - 140 121 | 2<sup>12</sup> | **7*2<sup>9</sup>** | 2<sup>11</sup> | 7*2<sup>8</sup> | x1.143 | --original probably faster |
| 140 122 - 157 293 | 2<sup>12</sup> | **63*2<sup>6</sup>** | 2<sup>11</sup> | 63*2<sup>5</sup> | x1.016 | --original probably faster |
| 157 294 - 159 744 | 2<sup>12</sup> | **2<sup>12</sup>** | 2<sup>11</sup> | 2<sup>11</sup> | x1.000 | --original / same size |
| 159 745 - 179 320 | 2<sup>13</sup> | **9*2<sup>9</sup>** | 2<sup>12</sup> | 9*2<sup>8</sup> | x1.778 | mixed probably faster |
| 179 321 - 208 609 | 2<sup>13</sup> | **21*2<sup>8</sup>** | 2<sup>12</sup> | 21*2<sup>7</sup> | x1.524 | mixed probably faster |
| 208 610 - 237 818 | 2<sup>13</sup> | **3*2<sup>11</sup>** | 2<sup>12</sup> | 3*2<sup>10</sup> | x1.333 | mixed probably faster |
| 237 819 - 276 658 | 2<sup>13</sup> | **7*2<sup>10</sup>** | 2<sup>12</sup> | 7*2<sup>9</sup> | x1.143 | --original probably faster |
| 276 659 - 310 555 | 2<sup>13</sup> | **63*2<sup>7</sup>** | 2<sup>12</sup> | 63*2<sup>6</sup> | x1.016 | --original probably faster |
| 310 556 - 315 392 | 2<sup>13</sup> | **2<sup>13</sup>** | 2<sup>12</sup> | 2<sup>12</sup> | x1.000 | --original / same size |
| 315 393 - 354 032 | 2<sup>14</sup> | **9*2<sup>10</sup>** | 2<sup>13</sup> | 9*2<sup>9</sup> | x1.778 | mixed probably faster |
| 354 033 - 411 842 | 2<sup>14</sup> | **21*2<sup>9</sup>** | 2<sup>13</sup> | 21*2<sup>8</sup> | x1.524 | mixed probably faster |
| 411 843 - 469 493 | 2<sup>14</sup> | **3*2<sup>12</sup>** | 2<sup>13</sup> | 3*2<sup>11</sup> | x1.333 | mixed probably faster |
| 469 494 - 546 148 | 2<sup>14</sup> | **7*2<sup>11</sup>** | 2<sup>13</sup> | 7*2<sup>10</sup> | x1.143 | --original probably faster |
| 546 149 - 613 047 | 2<sup>14</sup> | **63*2<sup>8</sup>** | 2<sup>13</sup> | 63*2<sup>7</sup> | x1.016 | --original probably faster |
| 613 048 - 622 592 | 2<sup>14</sup> | **2<sup>14</sup>** | 2<sup>13</sup> | 2<sup>13</sup> | x1.000 | --original / same size |
| 622 593 - 698 849 | 2<sup>15</sup> | **9*2<sup>11</sup>** | 2<sup>14</sup> | 9*2<sup>10</sup> | x1.778 | mixed probably faster |
| 698 850 - 812 933 | 2<sup>15</sup> | **21*2<sup>10</sup>** | 2<sup>14</sup> | 21*2<sup>9</sup> | x1.524 | mixed probably faster |
| 812 934 - 926 699 | 2<sup>15</sup> | **3*2<sup>13</sup>** | 2<sup>14</sup> | 3*2<sup>12</sup> | x1.333 | mixed probably faster |
| 926 700 - 1 077 961 | 2<sup>15</sup> | **7*2<sup>12</sup>** | 2<sup>14</sup> | 7*2<sup>11</sup> | x1.143 | --original probably faster |
| 1 077 962 - 1 209 966 | 2<sup>15</sup> | **63*2<sup>9</sup>** | 2<sup>14</sup> | 63*2<sup>8</sup> | x1.016 | --original probably faster |
| 1 209 967 - 1 228 800 | 2<sup>15</sup> | **2<sup>15</sup>** | 2<sup>14</sup> | 2<sup>14</sup> | x1.000 | --original / same size |
| 1 228 801 - 1 379 267 | 2<sup>16</sup> | **9*2<sup>12</sup>** | 2<sup>15</sup> | 9*2<sup>11</sup> | x1.778 | mixed probably faster |
| 1 379 268 - 1 604 363 | 2<sup>16</sup> | **21*2<sup>11</sup>** | 2<sup>15</sup> | 21*2<sup>10</sup> | x1.524 | mixed probably faster |
| 1 604 364 - 1 828 823 | 2<sup>16</sup> | **3*2<sup>14</sup>** | 2<sup>15</sup> | 3*2<sup>13</sup> | x1.333 | mixed probably faster |
| 1 828 824 - 2 127 251 | 2<sup>16</sup> | **7*2<sup>13</sup>** | 2<sup>15</sup> | 7*2<sup>12</sup> | x1.143 | --original probably faster |
| 2 127 252 - 2 387 676 | 2<sup>16</sup> | **63*2<sup>10</sup>** | 2<sup>15</sup> | 63*2<sup>9</sup> | x1.016 | --original probably faster |
| 2 387 677 - 2 424 832 | 2<sup>16</sup> | **2<sup>16</sup>** | 2<sup>15</sup> | 2<sup>15</sup> | x1.000 | --original / same size |
| 2 424 833 - 2 721 671 | 2<sup>17</sup> | **9*2<sup>13</sup>** | 2<sup>16</sup> | 9*2<sup>12</sup> | x1.778 | mixed probably faster |
| 2 721 672 - 3 165 719 | 2<sup>17</sup> | **21*2<sup>12</sup>** | 2<sup>16</sup> | 21*2<sup>11</sup> | x1.524 | mixed probably faster |
| 3 165 720 - 3 608 495 | 2<sup>17</sup> | **3*2<sup>15</sup>** | 2<sup>16</sup> | 3*2<sup>14</sup> | x1.333 | mixed probably faster |
| 3 608 496 - 4 197 159 | 2<sup>17</sup> | **7*2<sup>14</sup>** | 2<sup>16</sup> | 7*2<sup>13</sup> | x1.143 | --original probably faster |
| 4 197 160 - 4 710 841 | 2<sup>17</sup> | **63*2<sup>11</sup>** | 2<sup>16</sup> | 63*2<sup>10</sup> | x1.016 | --original probably faster |
| 4 710 842 - 4 784 128 | 2<sup>17</sup> | **2<sup>17</sup>** | 2<sup>16</sup> | 2<sup>16</sup> | x1.000 | --original / same size |
| 4 784 129 - 5 369 615 | 2<sup>18</sup> | **9*2<sup>14</sup>** | 2<sup>17</sup> | 9*2<sup>13</sup> | x1.778 | mixed probably faster |
| 5 369 616 - 6 245 422 | 2<sup>18</sup> | **21*2<sup>13</sup>** | 2<sup>17</sup> | 21*2<sup>12</sup> | x1.524 | mixed probably faster |
| 6 245 423 - 7 118 687 | 2<sup>18</sup> | **3*2<sup>16</sup>** | 2<sup>17</sup> | 3*2<sup>15</sup> | x1.333 | mixed probably faster |
| 7 118 688 - 8 279 630 | 2<sup>18</sup> | **7*2<sup>15</sup>** | 2<sup>17</sup> | 7*2<sup>14</sup> | x1.143 | --original probably faster |
| 8 279 631 - 9 292 659 | 2<sup>18</sup> | **63*2<sup>12</sup>** | 2<sup>17</sup> | 63*2<sup>11</sup> | x1.016 | --original probably faster |
| 9 292 660 - 9 437 184 | 2<sup>18</sup> | **2<sup>18</sup>** | 2<sup>17</sup> | 2<sup>17</sup> | x1.000 | --original / same size |
| 9 437 185 - 10 591 775 | 2<sup>19</sup> | **9*2<sup>15</sup>** | 2<sup>18</sup> | 9*2<sup>14</sup> | x1.778 | mixed probably faster |
| 10 591 776 - 12 318 812 | 2<sup>19</sup> | **21*2<sup>14</sup>** | 2<sup>18</sup> | 21*2<sup>13</sup> | x1.524 | mixed probably faster |
| 12 318 813 - 14 040 767 | 2<sup>19</sup> | **3*2<sup>17</sup>** | 2<sup>18</sup> | 3*2<sup>16</sup> | x1.333 | mixed probably faster |
| 14 040 768 - 16 329 884 | 2<sup>19</sup> | **7*2<sup>16</sup>** | 2<sup>18</sup> | 7*2<sup>15</sup> | x1.143 | --original probably faster |
| 16 329 885 - 18 327 270 | 2<sup>19</sup> | **63*2<sup>13</sup>** | 2<sup>18</sup> | 63*2<sup>12</sup> | x1.016 | --original probably faster |
| 18 327 271 - 18 612 224 | 2<sup>19</sup> | **2<sup>19</sup>** | 2<sup>18</sup> | 2<sup>18</sup> | x1.000 | --original / same size |
| 18 612 225 - 20 888 639 | 2<sup>20</sup> | **9*2<sup>16</sup>** | 2<sup>19</sup> | 9*2<sup>15</sup> | x1.778 | mixed probably faster |
| 20 888 640 - 24 293 561 | 2<sup>20</sup> | **21*2<sup>15</sup>** | 2<sup>19</sup> | 21*2<sup>14</sup> | x1.524 | mixed probably faster |
| 24 293 562 - 27 688 319 | 2<sup>20</sup> | **3*2<sup>18</sup>** | 2<sup>19</sup> | 3*2<sup>17</sup> | x1.333 | mixed probably faster |
| 27 688 320 - 32 201 016 | 2<sup>20</sup> | **7*2<sup>17</sup>** | 2<sup>19</sup> | 7*2<sup>16</sup> | x1.143 | --original probably faster |
| 32 201 017 - 36 138 445 | 2<sup>20</sup> | **63*2<sup>14</sup>** | 2<sup>19</sup> | 63*2<sup>13</sup> | x1.016 | --original probably faster |
| 36 138 446 - 36 700 160 | 2<sup>20</sup> | **2<sup>20</sup>** | 2<sup>19</sup> | 2<sup>19</sup> | x1.000 | --original / same size |
| 36 700 161 - 41 187 454 | 2<sup>21</sup> | **9*2<sup>17</sup>** | 2<sup>20</sup> | 9*2<sup>16</sup> | x1.778 | mixed probably faster |
| 41 187 455 - 47 898 995 | 2<sup>21</sup> | **21*2<sup>16</sup>** | 2<sup>20</sup> | 21*2<sup>15</sup> | x1.524 | mixed probably faster |
| 47 898 996 - 54 590 206 | 2<sup>21</sup> | **3*2<sup>19</sup>** | 2<sup>20</sup> | 3*2<sup>18</sup> | x1.333 | mixed probably faster |
| 54 590 207 - 63 484 528 | 2<sup>21</sup> | **7*2<sup>18</sup>** | 2<sup>20</sup> | 7*2<sup>17</sup> | x1.143 | --original probably faster |
| 63 484 529 - 71 244 699 | 2<sup>21</sup> | **63*2<sup>15</sup>** | 2<sup>20</sup> | 63*2<sup>14</sup> | x1.016 | --original probably faster |
| 71 244 700 - 72 351 744 | 2<sup>21</sup> | **2<sup>21</sup>** | 2<sup>20</sup> | 2<sup>20</sup> | x1.000 | --original / same size |
| 72 351 745 - 81 195 260 | 2<sup>22</sup> | **9*2<sup>18</sup>** | 2<sup>21</sup> | 9*2<sup>17</sup> | x1.778 | mixed probably faster |
| 81 195 261 - 94 421 734 | 2<sup>22</sup> | **21*2<sup>17</sup>** | 2<sup>21</sup> | 21*2<sup>16</sup> | x1.524 | mixed probably faster |
| 94 421 735 - 107 607 549 | 2<sup>22</sup> | **3*2<sup>20</sup>** | 2<sup>21</sup> | 3*2<sup>19</sup> | x1.333 | mixed probably faster |
| 107 607 550 - 125 134 049 | 2<sup>22</sup> | **7*2<sup>19</sup>** | 2<sup>21</sup> | 7*2<sup>18</sup> | x1.143 | --original probably faster |
| 125 134 050 - 140 425 014 | 2<sup>22</sup> | **63*2<sup>16</sup>** | 2<sup>21</sup> | 63*2<sup>15</sup> | x1.016 | --original probably faster |
| 140 425 015 - 142 606 336 | 2<sup>22</sup> | **2<sup>22</sup>** | 2<sup>21</sup> | 2<sup>21</sup> | x1.000 | --original / same size |
| 142 606 337 - 160 031 224 | 2<sup>23</sup> | **9*2<sup>19</sup>** | 2<sup>22</sup> | 9*2<sup>18</sup> | x1.778 | mixed probably faster |
| 160 031 225 - 186 090 957 | 2<sup>23</sup> | **21*2<sup>18</sup>** | 2<sup>22</sup> | 21*2<sup>17</sup> | x1.524 | mixed probably faster |
| 186 090 958 - 212 069 371 | 2<sup>23</sup> | **3*2<sup>21</sup>** | 2<sup>22</sup> | 3*2<sup>20</sup> | x1.333 | mixed probably faster |
| 212 069 372 - 246 598 082 | 2<sup>23</sup> | **7*2<sup>20</sup>** | 2<sup>22</sup> | 7*2<sup>19</sup> | x1.143 | --original probably faster |
| 246 598 083 - 276 721 261 | 2<sup>23</sup> | **63*2<sup>17</sup>** | 2<sup>22</sup> | 63*2<sup>16</sup> | x1.016 | --original probably faster |
| 276 721 262 - 281 018 368 | 2<sup>23</sup> | **2<sup>23</sup>** | 2<sup>22</sup> | 2<sup>22</sup> | x1.000 | --original / same size |
| 281 018 369 - 315 343 857 | 2<sup>24</sup> | **9*2<sup>20</sup>** | 2<sup>23</sup> | 9*2<sup>19</sup> | x1.778 | mixed probably faster |
