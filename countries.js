export const COUNTRIES = [
  {
    name: 'United States',
    iso2: 'US',
    capital: 'Washington, D.C.',
    lat: 38.9072,
    lon: -77.0369,
    gdp: 25462,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Finance & Real Estate', value: 5200 },
          { name: 'Professional Services', value: 3100 },
          { name: 'Healthcare', value: 2500 },
          { name: 'Information & Tech', value: 2300 },
          { name: 'Retail & Hospitality', value: 2100 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Manufacturing', value: 2500 },
          { name: 'Construction', value: 950 },
          { name: 'Energy & Utilities', value: 780 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Crops', value: 215 },
          { name: 'Livestock', value: 185 }
        ]
      }
    ]
  },
  {
    name: 'China',
    iso2: 'CN',
    capital: 'Beijing',
    lat: 39.9042,
    lon: 116.4074,
    gdp: 17960,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Wholesale & Retail', value: 2600 },
          { name: 'Financial Services', value: 1700 },
          { name: 'Transport & Logistics', value: 1400 },
          { name: 'ICT & R&D', value: 1300 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Manufacturing', value: 5200 },
          { name: 'Construction', value: 1800 },
          { name: 'Mining', value: 600 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Grains', value: 560 },
          { name: 'Livestock', value: 420 }
        ]
      }
    ]
  },
  {
    name: 'Japan',
    iso2: 'JP',
    capital: 'Tokyo',
    lat: 35.6762,
    lon: 139.6503,
    gdp: 4231,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Finance & Insurance', value: 780 },
          { name: 'Retail & Hospitality', value: 620 },
          { name: 'Transportation', value: 410 },
          { name: 'Government & Education', value: 460 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Manufacturing', value: 990 },
          { name: 'Construction', value: 260 },
          { name: 'Energy & Utilities', value: 180 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Rice & Grains', value: 70 },
          { name: 'Fisheries', value: 55 }
        ]
      }
    ]
  },
  {
    name: 'Germany',
    iso2: 'DE',
    capital: 'Berlin',
    lat: 52.52,
    lon: 13.405,
    gdp: 4256,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Business & Professional', value: 810 },
          { name: 'Finance & Real Estate', value: 460 },
          { name: 'Trade & Hospitality', value: 430 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Manufacturing', value: 1170 },
          { name: 'Automotive', value: 350 },
          { name: 'Construction', value: 250 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Crop Production', value: 45 },
          { name: 'Animal Husbandry', value: 35 }
        ]
      }
    ]
  },
  {
    name: 'India',
    iso2: 'IN',
    capital: 'New Delhi',
    lat: 28.6139,
    lon: 77.209,
    gdp: 3385,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'IT & Business Services', value: 580 },
          { name: 'Finance & Real Estate', value: 430 },
          { name: 'Trade & Tourism', value: 370 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Manufacturing', value: 710 },
          { name: 'Construction', value: 290 },
          { name: 'Energy', value: 150 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Crops', value: 310 },
          { name: 'Livestock', value: 190 }
        ]
      }
    ]
  },
  {
    name: 'United Kingdom',
    iso2: 'GB',
    capital: 'London',
    lat: 51.5072,
    lon: -0.1276,
    gdp: 3250,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Finance & Insurance', value: 670 },
          { name: 'Professional & Scientific', value: 490 },
          { name: 'Public Services', value: 410 },
          { name: 'Retail & Hospitality', value: 360 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Manufacturing', value: 360 },
          { name: 'Construction', value: 190 },
          { name: 'Energy', value: 95 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Agriculture & Fishing', value: 30 }
        ]
      }
    ],
    aliases: ['UK', 'Great Britain']
  },
  {
    name: 'France',
    iso2: 'FR',
    capital: 'Paris',
    lat: 48.8566,
    lon: 2.3522,
    gdp: 2930,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Public Administration', value: 470 },
          { name: 'Finance & Real Estate', value: 400 },
          { name: 'Trade & Transport', value: 420 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Manufacturing', value: 550 },
          { name: 'Construction', value: 180 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Wine & Specialty Crops', value: 55 },
          { name: 'Livestock', value: 40 }
        ]
      }
    ]
  },
  {
    name: 'Brazil',
    iso2: 'BR',
    capital: 'Brasília',
    lat: -15.8267,
    lon: -47.9218,
    gdp: 1924,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Retail & Hospitality', value: 430 },
          { name: 'Finance & Real Estate', value: 310 },
          { name: 'Public Services', value: 220 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Manufacturing', value: 410 },
          { name: 'Mining & Energy', value: 210 },
          { name: 'Construction', value: 150 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Soy & Grains', value: 140 },
          { name: 'Livestock', value: 120 },
          { name: 'Forestry', value: 60 }
        ]
      }
    ]
  },
  {
    name: 'Canada',
    iso2: 'CA',
    capital: 'Ottawa',
    lat: 45.4215,
    lon: -75.6972,
    gdp: 2163,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Finance & Real Estate', value: 430 },
          { name: 'Public Services', value: 310 },
          { name: 'Trade & Transport', value: 290 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Energy & Mining', value: 310 },
          { name: 'Manufacturing', value: 290 },
          { name: 'Construction', value: 150 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Prairie Grains', value: 55 },
          { name: 'Livestock', value: 45 }
        ]
      }
    ]
  },
  {
    name: 'Australia',
    iso2: 'AU',
    capital: 'Canberra',
    lat: -35.2809,
    lon: 149.13,
    gdp: 1717,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Professional & Scientific', value: 240 },
          { name: 'Finance & Real Estate', value: 300 },
          { name: 'Education & Health', value: 290 },
          { name: 'Tourism & Retail', value: 210 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Mining & Resources', value: 260 },
          { name: 'Manufacturing', value: 120 },
          { name: 'Construction', value: 150 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Livestock', value: 45 },
          { name: 'Crops & Viticulture', value: 35 }
        ]
      }
    ]
  },
  {
    name: 'Russia',
    iso2: 'RU',
    capital: 'Moscow',
    lat: 55.7558,
    lon: 37.6173,
    gdp: 2240,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Public Services', value: 370 },
          { name: 'Trade & Hospitality', value: 300 },
          { name: 'Finance & Real Estate', value: 220 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Oil & Gas', value: 620 },
          { name: 'Manufacturing', value: 350 },
          { name: 'Mining & Metals', value: 260 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Grains', value: 110 },
          { name: 'Livestock', value: 90 }
        ]
      }
    ]
  },
  {
    name: 'South Korea',
    iso2: 'KR',
    capital: 'Seoul',
    lat: 37.5665,
    lon: 126.978,
    gdp: 1741,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'ICT & Digital', value: 260 },
          { name: 'Finance & Real Estate', value: 210 },
          { name: 'Retail & Hospitality', value: 190 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Electronics', value: 290 },
          { name: 'Automotive', value: 220 },
          { name: 'Shipbuilding', value: 120 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Rice', value: 35 },
          { name: 'Livestock', value: 28 }
        ]
      }
    ]
  },
  {
    name: 'Mexico',
    iso2: 'MX',
    capital: 'Mexico City',
    lat: 19.4326,
    lon: -99.1332,
    gdp: 1469,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Trade & Hospitality', value: 310 },
          { name: 'Finance & Real Estate', value: 210 },
          { name: 'Public Services', value: 170 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Manufacturing', value: 370 },
          { name: 'Oil & Gas', value: 160 },
          { name: 'Construction', value: 130 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Crops', value: 90 },
          { name: 'Livestock', value: 70 }
        ]
      }
    ]
  },
  {
    name: 'Italy',
    iso2: 'IT',
    capital: 'Rome',
    lat: 41.9028,
    lon: 12.4964,
    gdp: 2100,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Tourism & Hospitality', value: 320 },
          { name: 'Finance & Real Estate', value: 360 },
          { name: 'Public Services', value: 300 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Manufacturing', value: 520 },
          { name: 'Luxury Goods', value: 190 },
          { name: 'Construction', value: 160 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Mediterranean Crops', value: 70 },
          { name: 'Wine & Olive Oil', value: 60 }
        ]
      }
    ]
  },
  {
    name: 'Spain',
    iso2: 'ES',
    capital: 'Madrid',
    lat: 40.4168,
    lon: -3.7038,
    gdp: 1515,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Tourism & Hospitality', value: 280 },
          { name: 'Finance & Real Estate', value: 200 },
          { name: 'Public Services', value: 220 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Manufacturing', value: 310 },
          { name: 'Construction', value: 140 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Citrus & Produce', value: 65 },
          { name: 'Viticulture', value: 50 }
        ]
      }
    ]
  },
  {
    name: 'South Africa',
    iso2: 'ZA',
    capital: 'Pretoria',
    lat: -25.7479,
    lon: 28.2293,
    gdp: 405,
    breakdown: [
      {
        name: 'Services',
        children: [
          { name: 'Finance & Business', value: 110 },
          { name: 'Trade & Hospitality', value: 90 },
          { name: 'Government & Education', value: 75 }
        ]
      },
      {
        name: 'Industry',
        children: [
          { name: 'Mining', value: 60 },
          { name: 'Manufacturing', value: 55 },
          { name: 'Energy', value: 35 }
        ]
      },
      {
        name: 'Agriculture',
        children: [
          { name: 'Crops', value: 18 },
          { name: 'Livestock', value: 22 }
        ]
      }
    ]
  }
]
