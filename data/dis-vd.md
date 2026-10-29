# DIS-VD: images per category

DIS5K validation split (DIS-VD): **470 images**, each with a ground-truth mask (`im/` and `gt/`, all paired), in **22 groups** and **164 sub-categories**.
The category comes from the filename: `<group id>#<group>#<sub id>#<sub-category>#<flickr id>.png`.

**Note for E1:** DIS-VD has no "person" group. The only fashion-related group is Accessories (20 images: bag, clothes, eyeglasses, jewelry, ring, shoe, watch). DIS-VD tests thin or complex structures (cables, chairs, scaffolds). It does not test people or hair, so we need PPM-100 for those.

## Per group

| # | Group | Sub-categories | Images |
|---:|---|---:|---:|
| 1 | Accessories | 7 | 20 |
| 2 | Aircraft | 7 | 18 |
| 3 | Aquatic | 5 | 11 |
| 4 | Architecture | 16 | 70 |
| 5 | Artifact | 5 | 14 |
| 6 | Automobile | 11 | 22 |
| 7 | Electrical | 8 | 33 |
| 8 | Electronics | 8 | 14 |
| 9 | Entertainment | 3 | 13 |
| 10 | Frame | 6 | 19 |
| 11 | Furniture | 15 | 63 |
| 12 | Graphics | 2 | 6 |
| 13 | Insect | 8 | 19 |
| 14 | Kitchenware | 8 | 21 |
| 15 | Machine | 3 | 5 |
| 16 | Music Instrument | 5 | 10 |
| 17 | Non-motor Vehicle | 10 | 34 |
| 18 | Plant | 2 | 2 |
| 19 | Ship | 5 | 9 |
| 20 | Sports | 10 | 15 |
| 21 | Tool | 15 | 40 |
| 22 | Weapon | 5 | 12 |
| | **Total** | **164** | **470** |

## Per sub-category

| Group | Sub-category | Images |
|---|---|---:|
| Accessories | Bag | 3 |
| Accessories | Clothes | 1 |
| Accessories | Eyeglasses | 3 |
| Accessories | Jewelry | 8 |
| Accessories | Ring | 3 |
| Accessories | Shoe | 1 |
| Accessories | Watch | 1 |
| Aircraft | Airplane | 7 |
| Aircraft | Balloon | 1 |
| Aircraft | HangGlider | 3 |
| Aircraft | Helicopter | 3 |
| Aircraft | Kite | 2 |
| Aircraft | Parachute | 1 |
| Aircraft | UAV | 1 |
| Aquatic | Lobster | 2 |
| Aquatic | Seadragon | 4 |
| Aquatic | Seahorse | 1 |
| Aquatic | Seal | 1 |
| Aquatic | Shrimp | 3 |
| Architecture | Bridge | 5 |
| Architecture | Crack | 3 |
| Architecture | Fence | 5 |
| Architecture | GasStation | 5 |
| Architecture | Gate | 5 |
| Architecture | ObservationTower | 4 |
| Architecture | OilWell | 1 |
| Architecture | Pavilion | 4 |
| Architecture | Pipe | 3 |
| Architecture | Scaffold | 7 |
| Architecture | Sculpture | 13 |
| Architecture | Stair | 3 |
| Architecture | WaterTower | 1 |
| Architecture | Well | 2 |
| Architecture | WindTurbine | 3 |
| Architecture | Windmill | 6 |
| Artifact | Basket | 3 |
| Artifact | Cage | 4 |
| Artifact | Handcraft | 3 |
| Artifact | Medal | 1 |
| Artifact | Toy | 3 |
| Automobile | BoomLift | 1 |
| Automobile | CableLift | 1 |
| Automobile | Car | 3 |
| Automobile | ConcretePumpTruck | 2 |
| Automobile | Crane | 1 |
| Automobile | Excavator | 1 |
| Automobile | Forklift | 4 |
| Automobile | Motorcycle | 2 |
| Automobile | SteeringWheel | 1 |
| Automobile | Tractor | 2 |
| Automobile | Truck | 4 |
| Electrical | Cable | 14 |
| Electrical | CeilingLamp | 4 |
| Electrical | FloorLamp | 6 |
| Electrical | Lightning | 1 |
| Electrical | StreetLamp | 5 |
| Electrical | TrafficLight | 1 |
| Electrical | TransmissionTower | 1 |
| Electrical | VacuumCleaner | 1 |
| Electronics | Antenna | 1 |
| Electronics | Camcorder | 2 |
| Electronics | Earphone | 2 |
| Electronics | Headset | 1 |
| Electronics | Microphone | 1 |
| Electronics | Monitor | 1 |
| Electronics | MusicPlayer | 2 |
| Electronics | Robot | 4 |
| Entertainment | GymEquipment | 5 |
| Entertainment | KidsPlayground | 3 |
| Entertainment | Swing | 5 |
| Frame | BasketballHoop | 2 |
| Frame | BicycleFrame | 1 |
| Frame | BicycleStand | 2 |
| Frame | MagazineRack | 1 |
| Frame | Rack | 11 |
| Frame | Skeleton | 2 |
| Furniture | Bench | 6 |
| Furniture | BunkBed | 3 |
| Furniture | Chair | 11 |
| Furniture | CoatHanger | 1 |
| Furniture | DentalChair | 2 |
| Furniture | Desk | 4 |
| Furniture | DeskChair | 2 |
| Furniture | Easel | 3 |
| Furniture | Hammock | 2 |
| Furniture | Handrail | 6 |
| Furniture | IroningTable | 3 |
| Furniture | Ladder | 10 |
| Furniture | Stool | 1 |
| Furniture | Table | 4 |
| Furniture | TableChair | 5 |
| Graphics | Diagram | 5 |
| Graphics | TrafficSign | 1 |
| Insect | Ant | 3 |
| Insect | Butterfly | 1 |
| Insect | Centipede | 5 |
| Insect | Dragonfly | 2 |
| Insect | Locust | 3 |
| Insect | Mantis | 1 |
| Insect | Mosquito | 3 |
| Insect | Spider | 1 |
| Kitchenware | Cup | 2 |
| Kitchenware | DishRack | 1 |
| Kitchenware | Hydrovalve | 6 |
| Kitchenware | Kitchenware | 4 |
| Kitchenware | Knife | 1 |
| Kitchenware | Stove | 1 |
| Kitchenware | WineGlass | 4 |
| Kitchenware | WireWhisk | 2 |
| Machine | Gear | 1 |
| Machine | Machine | 2 |
| Machine | PlowHarrow | 2 |
| Music Instrument | Drum | 1 |
| Music Instrument | MusicStand | 3 |
| Music Instrument | Piano | 2 |
| Music Instrument | Trombone | 3 |
| Music Instrument | Tuba | 1 |
| Non-motor Vehicle | BabyCarriage | 3 |
| Non-motor Vehicle | Bicycle | 8 |
| Non-motor Vehicle | Carriage | 3 |
| Non-motor Vehicle | Cart | 3 |
| Non-motor Vehicle | Handcart | 1 |
| Non-motor Vehicle | Mower | 2 |
| Non-motor Vehicle | Scooter | 2 |
| Non-motor Vehicle | ShoppingCart | 2 |
| Non-motor Vehicle | Tricycle | 5 |
| Non-motor Vehicle | Wheel | 5 |
| Plant | Bonsai | 1 |
| Plant | Tree | 1 |
| Ship | Boat | 2 |
| Ship | Canoe | 1 |
| Ship | Sailboat | 3 |
| Ship | Ship | 2 |
| Ship | SpeedBoat | 1 |
| Sports | Archery | 1 |
| Sports | Fencing | 1 |
| Sports | Fishing | 1 |
| Sports | PoleVault | 2 |
| Sports | Punchbag | 1 |
| Sports | Racket | 2 |
| Sports | RockClimbing | 2 |
| Sports | Skiing | 3 |
| Sports | TableTennis | 1 |
| Sports | Wakesurfing | 1 |
| Tool | Clock | 1 |
| Tool | Flag | 3 |
| Tool | Hook | 1 |
| Tool | Key | 6 |
| Tool | Rope | 1 |
| Tool | Scale | 2 |
| Tool | Scissor | 3 |
| Tool | Spade | 2 |
| Tool | Stand | 5 |
| Tool | Stationary | 2 |
| Tool | Sword | 4 |
| Tool | Tapeline | 1 |
| Tool | Tripod | 5 |
| Tool | Umbrella | 3 |
| Tool | Mop | 1 |
| Weapon | Cannon | 4 |
| Weapon | MachineGun | 2 |
| Weapon | Missile | 1 |
| Weapon | Rifle | 4 |
| Weapon | Tank | 1 |
