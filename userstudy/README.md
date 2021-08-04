# Setup

## pymongo

[Installation guide](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)

```bash
# add key for mongo
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 9DA31620334BD75D9DCB49F368818C72E52529D4
# This is for Ubuntu 18.04
echo "deb [ arch=amd64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.0.list
# update
sudo apt-get update
sudo apt-get install -y mongodb-org
```

### Start mongod

```bash
sudo service mongod start
```

And check the line of `/var/log/mongodb/mongod.log` for waiting for connection `[initandlisten] waiting for connections on port 27017`.

```bash
sudo service mongod stop
sudo service mongod restart
```

Mongo shell
```
mongo --host 127.0.0.1:27017
```

Video Stability
Total: 3200
Result
|type|choice A|choice B|
|:--|:--|:--|
| comb v.s. none | comb: 884    | none: 716     |
| comb v.s. flow | comb: 886    | flow: 714     |

Frame quality
Total: 3200
Result
|type|choice A|choice B|
|:--|:--|:--|
| comb v.s. none | comb: 752    | none: 848     |
| comb v.s. flow | comb: 839    | flow: 761     |

Frame quality
Total: 3200
Result
|type|choice A|choice B|
|:--|:--|:--|
| comb v.s. none | comb: 706    | none: 894     |
| comb v.s. flow | comb: 895    | flow: 705     |

Video stability
Total: 3200
Result
|type|choice A|choice B|
|:--|:--|:--|
| comb v.s. none | comb: 951    | none: 649     |
| comb v.s. flow | comb: 1008   | flow: 592     |

Video stability
Total: 1600
Result
|type|choice A|choice B|
|:--|:--|:--|
| flow v.s. none | flow: 872    | none: 728     |

starrynight
Total: 800
Result
|type|choice A|choice B|
|:--|:--|:--|
| comb v.s. none | comb: 253    | none: 147     |
| comb v.s. flow | comb: 248    | flow: 152     |
lamuse
Total: 800
Result
|type|choice A|choice B|
|:--|:--|:--|
| comb v.s. none | comb: 233    | none: 167     |
| comb v.s. flow | comb: 260    | flow: 140     |
feathers
Total: 800
Result
|type|choice A|choice B|
|:--|:--|:--|
| comb v.s. none | comb: 224    | none: 176     |
| comb v.s. flow | comb: 256    | flow: 144     |
composition
Total: 800
Result
|type|choice A|choice B|
|:--|:--|:--|
| comb v.s. none | comb: 241    | none: 159     |
| comb v.s. flow | comb: 244    | flow: 156     |

SFN Flow v.s. None

starrynight
Total: 400
Result
|type|choice A|choice B|
|:--|:--|:--|
| flow v.s. none | flow: 185    | none: 215     |
lamuse
Total: 400
Result
|type|choice A|choice B|
|:--|:--|:--|
| flow v.s. none | flow: 160    | none: 240     |
feathers
Total: 400
Result
|type|choice A|choice B|
|:--|:--|:--|
| flow v.s. none | flow: 164    | none: 236     |
composition
Total: 400
Result
|type|choice A|choice B|
|:--|:--|:--|
| flow v.s. none | flow: 186    | none: 214     |

Expr 1
Vote: 美观，稳定

Expr 2
Vote: 美观

|type|choice A|choice B|
|:--|:--|:--|
| tiger_composition.mp4 | flow: 11      | none: 9       |
| ambush_1_feathers.mp4 | flow: 11      | none: 9       |
| slackline_starrynight.mp4 | flow: 10  | none: 10      |
| tandem_composition.mp4 | flow: 9      | none: 11      |
| temple_1_lamuse.mp4 | flow: 8 | none: 12      |
| girl-dog_lamuse.mp4 | flow: 6 | none: 14      |
| PERTURBED_shaman_1_feathers.mp4 | flow: 11    | none: 9       |
| market_1_lamuse.mp4 | flow: 11        | none: 9       |
| man-bike_composition.mp4 | flow: 7    | none: 13      |
| cats-car_composition.mp4 | flow: 11   | none: 9       |
| bamboo_3_feathers.mp4 | flow: 5       | none: 15      |
| ambush_1_composition.mp4 | flow: 6    | none: 14      |
| gym_lamuse.mp4 | flow: 7      | none: 13      |
| cats-car_lamuse.mp4 | flow: 5 | none: 15      |
| horsejump-stick_lamuse.mp4 | flow: 5  | none: 15      |
| wall_starrynight.mp4 | flow: 7        | none: 13      |
| gym_composition.mp4 | flow: 9 | none: 11      |
| slackline_feathers.mp4 | flow: 13     | none: 7       |
| horsejump-stick_feathers.mp4 | flow: 6        | none: 14      |
| mountain_2_starrynight.mp4 | flow: 9  | none: 11      |
| guitar-violin_lamuse.mp4 | flow: 4    | none: 16      |
| wall_composition.mp4 | flow: 12       | none: 8       |
| bamboo_3_lamuse.mp4 | flow: 7 | none: 13      |
| cats-car_starrynight.mp4 | flow: 5    | none: 15      |
| cave_3_composition.mp4 | flow: 10     | none: 10      |
| helicopter_starrynight.mp4 | flow: 9  | none: 11      |
| slackline_lamuse.mp4 | flow: 9        | none: 11      |
| market_4_feathers.mp4 | flow: 7       | none: 13      |
| temple_1_composition.mp4 | flow: 10   | none: 10      |
| tandem_feathers.mp4 | flow: 6 | none: 14      |
| guitar-violin_feathers.mp4 | flow: 9  | none: 11      |
| subway_feathers.mp4 | flow: 13        | none: 7       |
| horsejump-stick_starrynight.mp4 | flow: 7     | none: 13      |
| subway_starrynight.mp4 | flow: 13     | none: 7       |
| cave_3_lamuse.mp4 | flow: 8   | none: 12      |
| subway_lamuse.mp4 | flow: 12  | none: 8       |
| mountain_2_feathers.mp4 | flow: 5     | none: 15      |
| subway_composition.mp4 | flow: 7      | none: 13      |
| market_4_starrynight.mp4 | flow: 10   | none: 10      |
| guitar-violin_composition.mp4 | flow: 8       | none: 12      |
| girl-dog_composition.mp4 | flow: 7    | none: 13      |
| tiger_lamuse.mp4 | flow: 5    | none: 15      |
| tandem_lamuse.mp4 | flow: 7   | none: 13      |
| man-bike_lamuse.mp4 | flow: 4 | none: 16      |
| tiger_starrynight.mp4 | flow: 8       | none: 12      |
| tandem_starrynight.mp4 | flow: 12     | none: 8       |
| PERTURBED_shaman_1_starrynight.mp4 | flow: 9  | none: 11      |
| mountain_2_lamuse.mp4 | flow: 9       | none: 11      |
| market_1_feathers.mp4 | flow: 8       | none: 12      |
| gym_feathers.mp4 | flow: 11   | none: 9       |
| mountain_2_composition.mp4 | flow: 12 | none: 8       |
| market_1_composition.mp4 | flow: 3    | none: 17      |
| cats-car_feathers.mp4 | flow: 7       | none: 13      |
| bamboo_3_starrynight.mp4 | flow: 10   | none: 10      |
| wall_lamuse.mp4 | flow: 13    | none: 7       |
| tiger_feathers.mp4 | flow: 12 | none: 8       |
| market_4_lamuse.mp4 | flow: 8 | none: 12      |
| cave_3_starrynight.mp4 | flow: 13     | none: 7       |
| girl-dog_starrynight.mp4 | flow: 7    | none: 13      |
| girl-dog_feathers.mp4 | flow: 11      | none: 9       |
| ambush_1_starrynight.mp4 | flow: 8    | none: 12      |
| cave_3_feathers.mp4 | flow: 9 | none: 11      |
| PERTURBED_shaman_1_lamuse.mp4 | flow: 6       | none: 14      |
| wall_feathers.mp4 | flow: 5   | none: 15      |
| temple_1_feathers.mp4 | flow: 4       | none: 16      |
| market_4_composition.mp4 | flow: 8    | none: 12      |
| man-bike_feathers.mp4 | flow: 8       | none: 12      |
| bamboo_3_composition.mp4 | flow: 12   | none: 8       |
| market_1_starrynight.mp4 | flow: 7    | none: 13      |
| helicopter_feathers.mp4 | flow: 8     | none: 12      |
| gym_starrynight.mp4 | flow: 6 | none: 14      |
| guitar-violin_starrynight.mp4 | flow: 9       | none: 11      |
| temple_1_starrynight.mp4 | flow: 12   | none: 8       |
| man-bike_starrynight.mp4 | flow: 8    | none: 12      |
| ambush_1_lamuse.mp4 | flow: 4 | none: 16      |
| helicopter_composition.mp4 | flow: 12 | none: 8       |
| slackline_composition.mp4 | flow: 8   | none: 12      |
| horsejump-stick_composition.mp4 | flow: 13    | none: 7       |
| PERTURBED_shaman_1_composition.mp4 | flow: 11 | none: 9       |
| helicopter_lamuse.mp4 | flow: 14      | none: 6       |